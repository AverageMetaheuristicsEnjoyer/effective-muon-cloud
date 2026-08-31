#!/usr/bin/env python3
"""Benchmark one dense, reference-Tucker, or optimized-Tucker training step."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import tempfile
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src", ROOT / "experiments/fused_persistent_tucker"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

import models.tucker_linear as tucker_linear_module  # noqa: E402
from experiments.fused_persistent_tucker.custom_backward.grouped_retraction import (  # noqa: E402
    grouped_retract_tucker_modules_,
)
from experiments.fused_persistent_tucker.custom_backward.integration import (  # noqa: E402
    install_custom_backward,
)
from experiments.fused_persistent_tucker.custom_backward.parallel_muon import (  # noqa: E402
    ParallelGroupedMuonLite,
)
from models.tucker_linear import TuckerLinear, retract_tucker_modules_  # noqa: E402
from models.utils import get_model  # noqa: E402
from scripts.tucker_benchmark.common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    FAST_ISO_FFN_WIDTHS,
    MODEL_SPECS,
    TUCKER_MODE_LAYOUTS,
    TUCKER_RANK_MULTIPLE,
    VARIANTS,
    _aligned_factor_pair,
    atomic_write_json,
    model_geometry,
    requested_controls,
    summarize,
    tucker_benchmark_plan,
    variant_spec,
)
from third_party.lite.muonlite import MuonLite  # noqa: E402


TUCKER_VARIANTS = {"tucker_reference", "tucker_parallel"}


def make_config(args) -> SimpleNamespace:
    is_tucker = args.variant in TUCKER_VARIANTS
    if is_tucker:
        geometry, rank_plan, tucker_parameters, _ = tucker_benchmark_plan(
            args.model_size, args.tucker_rank_profile, args.tucker_mode_layout
        )
    else:
        geometry = model_geometry(args.model_size)
        rank_plan = None
        tucker_parameters = geometry["dense_parameters"]
    return SimpleNamespace(
        model="llama",
        vocab_size=50_304,
        sequence_length=args.sequence_length,
        batch_size=args.microbatch,
        n_layer=geometry["n_layer"],
        n_embd=geometry["n_embd"],
        n_head=geometry["n_head"],
        dropout=0.0,
        bias=False,
        init_std=0.02,
        rmsnorm_eps=1e-5,
        multiple_of=256,
        ffn_hidden_size=geometry["intermediate_size"],
        label_smoothing=0.0,
        qkv_clipping=False,
        qkv_clipping_factor=1.0,
        attention_type="standard",
        linear_parameterization="tucker" if is_tucker else "dense",
        tucker_rank="auto",
        tucker_ranks=None,
        tucker_attention_ranks=None,
        tucker_gate_up_ranks=None,
        tucker_down_ranks=None,
        tucker_rank_plan=rank_plan,
        tucker_mode_layout=args.tucker_mode_layout,
        tucker_mode_multiple=(
            1
            if args.tucker_mode_layout != "balanced4"
            or args.tucker_rank_profile != "iso"
            or args.model_size in FAST_ISO_FFN_WIDTHS
            else TUCKER_RANK_MULTIPLE
        ),
        tucker_terms=1,
        tucker_equal_params=False,
        tucker_forward_mode="chunked_contract",
        tucker_contract_chunk_size=args.microbatch * args.sequence_length,
        tucker_head_contract_chunk_size=args.microbatch * args.sequence_length,
        tucker_dense_adamw_matrices=is_tucker,
        tucker_retract_every_step=is_tucker,
        tucker_vector_transport=False,
        tucker_riemannian_muon=False,
        target_parameter_count=(
            tucker_parameters if is_tucker else geometry["dense_parameters"]
        ),
        target_parameter_tolerance=0,
        dtype="bfloat16",
        device="cuda",
        liger_kernels=True,
        activation_checkpointing=False,
        fp8=False,
        fp8_optim=False,
        qargs=None,
    )


def instantiate_model(config: SimpleNamespace, device: torch.device):
    config.device = str(device)
    if device.type != "cuda":
        config.liger_kernels = False
    rank_plan = config.tucker_rank_plan
    original_factor_pair = tucker_linear_module.balanced_factor_pair
    try:
        if rank_plan is not None:
            if config.tucker_mode_multiple > 1:
                tucker_linear_module.balanced_factor_pair = lambda value: _aligned_factor_pair(
                    value, TUCKER_RANK_MULTIPLE
                )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json") as handle:
                json.dump({"module_ranks": rank_plan}, handle)
                handle.flush()
                config.tucker_rank_plan = handle.name
                with torch.device(device):
                    model = get_model(config)
        else:
            with torch.device(device):
                model = get_model(config)
    finally:
        config.tucker_rank_plan = rank_plan
        tucker_linear_module.balanced_factor_pair = original_factor_pair
    model.train()
    if any(parameter.dtype != torch.float32 for parameter in model.parameters()):
        raise RuntimeError("all model/master parameters must remain float32")
    return model


def parameter_groups(model, weight_decay: float) -> list[dict]:
    by_name = dict(model.named_parameters())
    groups = []
    for specification in model.get_parameter_group_specs():
        group = {key: value for key, value in specification.items() if key != "params"}
        group["params"] = [by_name[name] for name in specification["params"]]
        group.setdefault("weight_decay", weight_decay)
        groups.append(group)
    return groups


def build_muon_optimizer(args, model, *, parallel: bool):
    muon_params = []
    adamw_params = []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and not any(
            marker in name for marker in ("wte", "wpe", "lm_head", "embed", "core_logits")
        ):
            muon_params.append((name, parameter))
        else:
            adamw_params.append((name, parameter))

    optimizer_class = ParallelGroupedMuonLite if parallel else MuonLite
    kwargs = {}
    if parallel:
        kwargs.update(
            core_microbatch=args.tucker_muon_core_microbatch,
            parallel_streams=args.tucker_muon_streams,
        )
    optimizer = optimizer_class(
        muon_params=muon_params,
        adamw_params=adamw_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
        ns_steps=6,
        muon_theta=args.momentum,
        adamw_betas=(args.beta1, args.beta2),
        adamw_eps=args.eps,
        total_steps=1_000_000,
        warmup_steps=0,
        beta1=0.0,
        beta2=0.0,
        chi=1.0,
        chi_adamw=1.0,
        subspace_ratio=0.0,
        **kwargs,
    )
    split = {
        "muon": sum(parameter.numel() for _, parameter in muon_params),
        "adamw": sum(parameter.numel() for _, parameter in adamw_params),
    }
    if parallel:
        split["parallel_cores"] = optimizer.grouped_core_count
        split["parallel_factors"] = optimizer.grouped_factor_count
    return optimizer, split


def build_model_and_optimizer(args, device: torch.device):
    if args.variant == "tucker_parallel":
        install_custom_backward(cache_policy=args.tucker_cache_policy)
    config = make_config(args)
    model = instantiate_model(config, device)

    if args.variant in TUCKER_VARIANTS:
        modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]
        if len(modules) != config.n_layer * 7:
            raise RuntimeError(
                f"expected {config.n_layer * 7} internal Tucker modules, got {len(modules)}"
            )
        if not isinstance(model.lm_head, torch.nn.Linear):
            raise RuntimeError("lm_head must remain dense nn.Linear")
        forward_modes = {module.resolved_forward_mode for module in modules}
        if forward_modes != {"chunked_contract"}:
            raise RuntimeError(f"unexpected Tucker forward modes: {forward_modes}")

    if args.variant == "dense_adamw":
        groups = parameter_groups(model, args.weight_decay)
        for group in groups:
            group.pop("is_proj_params", None)
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            eps=args.eps,
            fused=True,
        )
        split = {"adamw": sum(parameter.numel() for parameter in model.parameters())}
        post_step = None
    elif args.variant == "dense_muon":
        optimizer, split = build_muon_optimizer(args, model, parallel=False)
        post_step = None
    elif args.variant == "tucker_reference":
        optimizer, split = build_muon_optimizer(args, model, parallel=False)

        def post_step():
            retract_tucker_modules_(
                model,
                optimizer=optimizer,
                transport_optimizer_state=False,
            )

    else:
        optimizer, split = build_muon_optimizer(args, model, parallel=True)

        def post_step():
            grouped_retract_tucker_modules_(
                model,
                optimizer=optimizer,
                transport_optimizer_state=False,
            )

    return model, optimizer, post_step, split


def tensor_bytes(value, seen: set[int] | None = None) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list)):
        items = value
    else:
        items = getattr(value, "__dict__", {}).values()
    return sum(tensor_bytes(item, seen) for item in items)


def tensor_dtypes(value) -> set[str]:
    if isinstance(value, torch.Tensor):
        return {str(value.dtype).removeprefix("torch.")}
    if isinstance(value, dict):
        items = value.values()
    elif isinstance(value, (tuple, list)):
        items = value
    else:
        items = getattr(value, "__dict__", {}).values()
    result = set()
    for item in items:
        result.update(tensor_dtypes(item))
    return result


def timed_step(
    model,
    optimizer,
    post_step,
    batches,
    stream,
    grad_clip: float,
    *,
    capture_memory: bool = False,
) -> dict:
    device = batches[0][0].device
    if capture_memory:
        torch.cuda.reset_peak_memory_stats(device)
    total_start = torch.cuda.Event(enable_timing=True)
    total_end = torch.cuda.Event(enable_timing=True)
    forward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    backward_pairs = [
        (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
        for _ in batches
    ]
    clip_start = torch.cuda.Event(enable_timing=True)
    clip_end = torch.cuda.Event(enable_timing=True)
    optimizer_start = torch.cuda.Event(enable_timing=True)
    optimizer_end = torch.cuda.Event(enable_timing=True)
    retraction_start = torch.cuda.Event(enable_timing=True)
    retraction_end = torch.cuda.Event(enable_timing=True)

    host_start = time.perf_counter_ns()
    losses = []
    with torch.cuda.stream(stream):
        total_start.record(stream)
        for index, (inputs, targets) in enumerate(batches):
            forward_pairs[index][0].record(stream)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = model(inputs, targets=targets)["loss"] / len(batches)
            forward_pairs[index][1].record(stream)
            backward_pairs[index][0].record(stream)
            loss.backward()
            backward_pairs[index][1].record(stream)
            losses.append(loss.detach())
        forward_backward_peak_bytes = (
            torch.cuda.max_memory_allocated(device) if capture_memory else None
        )

        clip_start.record(stream)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        clip_end.record(stream)
        optimizer_start.record(stream)
        optimizer.step()
        optimizer_end.record(stream)
        retraction_start.record(stream)
        if post_step is not None:
            post_step()
        retraction_end.record(stream)
        optimizer.zero_grad(set_to_none=True)
        total_end.record(stream)

    torch.cuda.synchronize()
    full_step_peak_bytes = torch.cuda.max_memory_allocated(device) if capture_memory else None
    return {
        "host_total_ms": (time.perf_counter_ns() - host_start) / 1e6,
        "gpu_total_ms": total_start.elapsed_time(total_end),
        "forward_ms": sum(start.elapsed_time(end) for start, end in forward_pairs),
        "backward_ms": sum(start.elapsed_time(end) for start, end in backward_pairs),
        "forward_backward_ms": sum(
            start.elapsed_time(end) for start, end in (*forward_pairs, *backward_pairs)
        ),
        "grad_clip_ms": clip_start.elapsed_time(clip_end),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "retraction_ms": retraction_start.elapsed_time(retraction_end),
        "forward_backward_peak_bytes": forward_backward_peak_bytes,
        "full_step_peak_bytes": full_step_peak_bytes,
        "loss": sum(float(loss.float()) for loss in losses),
    }


METRICS = (
    "host_total_ms",
    "gpu_total_ms",
    "forward_ms",
    "backward_ms",
    "forward_backward_ms",
    "grad_clip_ms",
    "optimizer_ms",
    "retraction_ms",
    "tokens_per_second",
    "tokens_per_second_forward_backward",
)


def run(args) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    variant = variant_spec(args.variant)
    if args.variant in TUCKER_VARIANTS:
        geometry, _, tucker_parameters, _ = tucker_benchmark_plan(
            args.model_size,
            args.tucker_rank_profile,
            args.tucker_mode_layout,
        )
    else:
        geometry = model_geometry(args.model_size)
        tucker_parameters = geometry["dense_parameters"]
    model, optimizer, post_step, optimizer_split = build_model_and_optimizer(args, device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    expected_parameters = (
        tucker_parameters if args.variant in TUCKER_VARIANTS else geometry["dense_parameters"]
    )
    if actual_parameters != expected_parameters:
        raise RuntimeError(
            f"parameter-count mismatch: actual={actual_parameters}, expected={expected_parameters}"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 17)
    batches = [
        (
            torch.randint(
                0,
                50_304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
            torch.randint(
                0,
                50_304,
                (args.microbatch, args.sequence_length),
                generator=generator,
                device=device,
            ),
        )
        for _ in range(args.accumulation_steps)
    ]
    tokens_per_step = args.microbatch * args.sequence_length * args.accumulation_steps
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    stream.synchronize()

    for _ in range(args.warmup_steps):
        sample = timed_step(model, optimizer, post_step, batches, stream, args.grad_clip)
        if not math.isfinite(sample["loss"]):
            raise RuntimeError(f"non-finite warmup loss: {sample['loss']}")

    samples = []
    for iteration in range(args.measured_steps):
        sample = timed_step(model, optimizer, post_step, batches, stream, args.grad_clip)
        if not math.isfinite(sample["loss"]):
            raise RuntimeError(f"non-finite measured loss at {iteration}: {sample['loss']}")
        sample["iteration"] = iteration
        sample["tokens_per_second"] = tokens_per_step / (sample["host_total_ms"] / 1000.0)
        sample["tokens_per_second_forward_backward"] = tokens_per_step / (
            sample["forward_backward_ms"] / 1000.0
        )
        samples.append(sample)

    memory_sample = timed_step(
        model,
        optimizer,
        post_step,
        batches,
        stream,
        args.grad_clip,
        capture_memory=True,
    )
    model_bytes = sum(tensor_bytes(parameter) for parameter in model.parameters())
    memory = {
        "model_bytes": model_bytes,
        "gradient_bytes_nominal": model_bytes,
        "optimizer_state_bytes": tensor_bytes(optimizer.state),
        "optimizer_state_dtypes": sorted(tensor_dtypes(optimizer.state)),
        "forward_backward_peak_allocated_bytes": memory_sample["forward_backward_peak_bytes"],
        "peak_allocated_bytes": memory_sample["full_step_peak_bytes"],
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    tucker_stats = getattr(model, "_tucker_replacement_stats", None)
    return {
        "status": "complete",
        "model": {
            "name": geometry["name"],
            "label": geometry["label"],
            "n_layer": geometry["n_layer"],
            "n_embd": geometry["n_embd"],
            "n_head": geometry["n_head"],
            "intermediate_size": geometry["intermediate_size"],
            "dense_parameters": geometry["dense_parameters"],
            "actual_parameters": actual_parameters,
            "parameter_difference_from_dense": actual_parameters - geometry["dense_parameters"],
            "tucker_rank_profile": (
                args.tucker_rank_profile if args.variant in TUCKER_VARIANTS else None
            ),
            "tucker_mode_layout": (
                args.tucker_mode_layout if args.variant in TUCKER_VARIANTS else None
            ),
            "dense_lm_head": isinstance(model.lm_head, torch.nn.Linear),
            "tucker_forward_modes": (
                dict(tucker_stats.forward_modes) if tucker_stats is not None else None
            ),
            "tucker_plans": (
                [
                    {"shape": shape, "ranks": ranks, "count": count}
                    for shape, ranks, _, count in tucker_stats.plans
                ]
                if tucker_stats is not None
                else None
            ),
        },
        "variant": variant,
        "benchmark": {
            **requested_controls(args, args.microbatch, args.accumulation_steps),
            "optimizer_backend": (
                "torch.optim.AdamW"
                if args.variant == "dense_adamw"
                else "ParallelGroupedMuonLite"
                if args.variant == "tucker_parallel"
                else "MuonLite"
            ),
            "adamw_fused": args.variant == "dense_adamw",
            "cuda_timing": "events on a dedicated stream; device-wide sync at step end",
            "optimizer_parameter_split": optimizer_split,
            "optimizer_ms_excludes_retraction": True,
        },
        "gpu": {
            "uuid": f"GPU-{torch.cuda.get_device_properties(device).uuid}",
            "logical_device": str(device),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "exclusive": args.exclusive_gpu,
        },
        "memory": memory,
        "summary": {
            metric: summarize([sample[metric] for sample in samples]) for metric in METRICS
        },
        "samples": samples,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "pid": os.getpid(),
        },
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True, choices=[item["name"] for item in VARIANTS])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--model-size", required=True, choices=[item["name"] for item in MODEL_SPECS])
    parser.add_argument("--tucker-rank-profile", default="iso")
    parser.add_argument(
        "--tucker-mode-layout", choices=TUCKER_MODE_LAYOUTS, default="balanced4"
    )
    parser.add_argument("--exclusive-gpu", action="store_true")
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--tucker-cache-policy",
        choices=("persistent", "recast", "hybrid_gate_up"),
        default="recast",
    )
    parser.add_argument("--tucker-muon-core-microbatch", type=int, default=1)
    parser.add_argument("--tucker-muon-streams", type=int, default=2)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.microbatch, args.sequence_length, args.accumulation_steps) <= 0:
        raise ValueError("microbatch, sequence length and accumulation steps must be positive")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured steps must be positive")
    if args.tucker_muon_core_microbatch < 1 or args.tucker_muon_streams < 1:
        raise ValueError("Tucker Muon microbatch and stream count must be positive")
    started = time.time()
    try:
        payload = run(args)
        payload["wall_started_unix"] = started
        payload["wall_finished_unix"] = time.time()
        atomic_write_json(args.output, payload)
        print(
            json.dumps(
                {
                    "status": payload["status"],
                    "model": args.model_size,
                    "variant": payload["variant"]["name"],
                    "microbatch": args.microbatch,
                    "median_ms": payload["summary"]["host_total_ms"]["median"],
                    "forward_backward_ms": payload["summary"]["forward_backward_ms"]["median"],
                    "optimizer_ms": payload["summary"]["optimizer_ms"]["median"],
                    "retraction_ms": payload["summary"]["retraction_ms"]["median"],
                    "peak_gb": payload["memory"]["peak_allocated_bytes"] / 1e9,
                },
                sort_keys=True,
            )
        )
    except BaseException as error:
        status = "oom" if isinstance(error, torch.cuda.OutOfMemoryError) else "failed"
        payload = {
            "status": status,
            "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
            "variant": args.variant,
            "requested_controls": requested_controls(
                args, args.microbatch, args.accumulation_steps
            ),
            "wall_started_unix": started,
            "wall_finished_unix": time.time(),
        }
        atomic_write_json(args.output, payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        raise
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
