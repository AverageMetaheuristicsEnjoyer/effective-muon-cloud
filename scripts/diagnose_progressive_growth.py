#!/usr/bin/env python3

import argparse
from contextlib import contextmanager
import gc
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src", ROOT / "experiments" / "fused_persistent_tucker"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from experiments.fused_persistent_tucker.custom_backward.integration import (
    install_custom_backward,
)

install_custom_backward(cache_policy="recast")

from main import get_args
from models.tucker_linear import TuckerLinear
from models.utils import get_model
from optim.progressive_tucker import (
    _tucker_modules,
    build_progressive_rank_stages,
    expand_tucker_model_to_plan_,
    parse_progressive_stages,
    restore_progressive_tucker_shapes_,
)


@contextmanager
def tf32(enabled):
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = enabled
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def model_args(architecture):
    original = sys.argv
    try:
        sys.argv = [original[0], "--experiment-name", "growth-diagnostic"]
        args = get_args()
    finally:
        sys.argv = original
    for key, value in architecture.items():
        setattr(args, key, value)
    args.device = "cuda:0"
    args.qargs = None
    return args


@torch.inference_mode()
def logits(model, tokens, *, bf16):
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        return model(tokens, get_logits=True)["logits"].float().cpu()


def tensor_error(before, after):
    delta = after - before
    return {
        "relative": float(
            (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(before).clamp_min(1e-30)).cpu()
        ),
        "max_absolute": float(delta.abs().max().cpu()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    args = parser.parse_args()

    main_path = args.checkpoint / "main.pt"
    worker_path = args.checkpoint / "worker_0.pt"
    print(
        json.dumps(
            {
                "event": "checkpoint_files",
                "main_bytes": main_path.stat().st_size,
                "worker_bytes": worker_path.stat().st_size,
            },
            sort_keys=True,
        )
    )

    checkpoint = torch.load(main_path, map_location="cpu", weights_only=False)
    worker = torch.load(worker_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint["itr"])
    progressive_state = checkpoint["progressive_tucker"]
    architecture = checkpoint["architecture"]
    if "train_reader_state" not in worker:
        raise RuntimeError("Checkpoint has no FineWeb reader state")
    print(
        json.dumps(
            {
                "event": "checkpoint_loaded",
                "iteration": iteration,
                "progressive_stage": int(progressive_state["stage_index"]),
                "optimizer_state_entries": len(checkpoint["optimizer"]["state"]),
                "reader_state": True,
            },
            sort_keys=True,
        )
    )

    cfg = model_args(architecture)
    torch.manual_seed(cfg.seed)
    model = get_model(cfg).to(cfg.device)
    restore_progressive_tucker_shapes_(model, None, progressive_state)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    del checkpoint, worker
    gc.collect()

    stages = build_progressive_rank_stages(
        model,
        parse_progressive_stages(architecture["tucker_progressive_stages"]),
    )
    stage_index = int(progressive_state["stage_index"])
    target = stages[stage_index + 1]
    if iteration >= target.step:
        raise RuntimeError(
            f"Checkpoint iteration {iteration} is not before growth step {target.step}"
        )
    print(
        json.dumps(
            {
                "event": "growth_target",
                "from_stage": stage_index,
                "to_stage": stage_index + 1,
                "growth_step": target.step,
                "target_parameters": target.target_parameters,
                "actual_parameters": target.actual_parameters,
            },
            sort_keys=True,
        )
    )

    modules = _tucker_modules(model)
    before_tf32 = {}
    before_fp32 = {}
    before_fp64 = {}
    old_parameters = {}
    for name, module in modules.items():
        with tf32(True):
            before_tf32[name] = module.materialize_weight(dtype=torch.float32).cpu()
        with tf32(False):
            before_fp32[name] = module.materialize_weight(dtype=torch.float32).cpu()
        before_fp64[name] = module.materialize_weight(dtype=torch.float64).cpu()
        old_parameters[name] = {
            "ranks": tuple(module.ranks),
            "core": module.core_matrix.detach().cpu().clone(),
            "factors": tuple(
                factor.detach().cpu().clone()
                for factor in (module.U1, module.U2, module.U3, module.U4)
            ),
        }

    generator = torch.Generator(device=cfg.device)
    generator.manual_seed(20260828)
    tokens = torch.randint(
        cfg.vocab_size,
        (2, 128),
        generator=generator,
        device=cfg.device,
    )
    with tf32(True):
        before_logits_tf32 = logits(model, tokens, bf16=False)
    with tf32(False):
        before_logits_fp32 = logits(model, tokens, bf16=False)
    before_logits_bf16 = logits(model, tokens, bf16=True)

    expansion = expand_tucker_model_to_plan_(
        model,
        None,
        target.ranks,
        seed=int(architecture["tucker_progressive_seed"]) + (stage_index + 1) * 1_000_003,
        verify_function=False,
    )

    layer_results = []
    parameter_copy_ok = True
    new_core_zero = True
    for name, module in modules.items():
        with tf32(True):
            after_tf32 = module.materialize_weight(dtype=torch.float32)
        with tf32(False):
            after32 = module.materialize_weight(dtype=torch.float32)
        after64 = module.materialize_weight(dtype=torch.float64)
        error_tf32 = tensor_error(before_tf32.pop(name).to(cfg.device), after_tf32)
        error32 = tensor_error(before_fp32.pop(name).to(cfg.device), after32)
        error64 = tensor_error(before_fp64.pop(name).to(cfg.device), after64)

        old = old_parameters.pop(name)
        old_ranks = old["ranks"]
        for factor, old_factor, old_rank in zip(
            (module.U1, module.U2, module.U3, module.U4),
            old["factors"],
            old_ranks,
        ):
            parameter_copy_ok &= torch.equal(
                factor[:, :old_rank], old_factor.to(cfg.device)
            )
        old_r1, old_r2, old_r3, old_r4 = old_ranks
        new_r1, new_r2, new_r3, new_r4 = module.ranks
        core = module.core_matrix.reshape(new_r3, new_r4, new_r1, new_r2)
        old_core = old["core"].reshape(old_r3, old_r4, old_r1, old_r2).to(cfg.device)
        parameter_copy_ok &= torch.equal(
            core[:old_r3, :old_r4, :old_r1, :old_r2], old_core
        )
        outside = core.detach().clone()
        outside[:old_r3, :old_r4, :old_r1, :old_r2] = 0
        new_core_zero &= int(torch.count_nonzero(outside).cpu()) == 0

        layer_results.append(
            {
                "name": name,
                "old_ranks": list(old_ranks),
                "new_ranks": list(module.ranks),
                "relative_tf32": error_tf32["relative"],
                "max_absolute_tf32": error_tf32["max_absolute"],
                "relative_fp32": error32["relative"],
                "max_absolute_fp32": error32["max_absolute"],
                "relative_fp64": error64["relative"],
                "max_absolute_fp64": error64["max_absolute"],
            }
        )
        del after_tf32, after32, after64, outside

    with tf32(True):
        after_logits_tf32 = logits(model, tokens, bf16=False)
    with tf32(False):
        after_logits_fp32 = logits(model, tokens, bf16=False)
    after_logits_bf16 = logits(model, tokens, bf16=True)
    logits_tf32 = tensor_error(before_logits_tf32, after_logits_tf32)
    logits_fp32 = tensor_error(before_logits_fp32, after_logits_fp32)
    logits_bf16 = tensor_error(before_logits_bf16, after_logits_bf16)
    logits_tf32["argmax_equal"] = bool(
        torch.equal(before_logits_tf32.argmax(dim=-1), after_logits_tf32.argmax(dim=-1))
    )
    logits_fp32["argmax_equal"] = bool(
        torch.equal(before_logits_fp32.argmax(dim=-1), after_logits_fp32.argmax(dim=-1))
    )
    logits_bf16["argmax_equal"] = bool(
        torch.equal(before_logits_bf16.argmax(dim=-1), after_logits_bf16.argmax(dim=-1))
    )

    threshold = float(architecture["tucker_progressive_verify_rtol"])
    failures = [row for row in layer_results if row["relative_tf32"] > threshold]
    print(json.dumps({"event": "expansion", **expansion}, sort_keys=True))
    print(
        json.dumps(
            {
                "event": "structural_checks",
                "old_parameters_bitwise_equal": parameter_copy_ok,
                "new_core_outside_old_block_zero": new_core_zero,
            },
            sort_keys=True,
        )
    )
    for row in sorted(layer_results, key=lambda item: item["relative_tf32"], reverse=True)[:12]:
        print(json.dumps({"event": "layer_error", **row}, sort_keys=True))
    print(
        json.dumps(
            {
                "event": "threshold_summary",
                "threshold": threshold,
                "layers_checked": len(layer_results),
                "layers_over_threshold": len(failures),
                "first_over_threshold": failures[0]["name"] if failures else None,
                "max_relative_tf32": max(row["relative_tf32"] for row in layer_results),
                "max_relative_fp32": max(row["relative_fp32"] for row in layer_results),
                "max_relative_fp64": max(row["relative_fp64"] for row in layer_results),
            },
            sort_keys=True,
        )
    )
    print(json.dumps({"event": "logits_tf32", **logits_tf32}, sort_keys=True))
    print(json.dumps({"event": "logits_fp32", **logits_fp32}, sort_keys=True))
    print(json.dumps({"event": "logits_bf16", **logits_bf16}, sort_keys=True))


if __name__ == "__main__":
    main()
