#!/usr/bin/env python3
"""Benchmark one complete 257M dense or static-Tucker training step."""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
for item in (ROOT, ROOT / "src"):
    if str(item) not in sys.path:
        sys.path.insert(0, str(item))

from models.tucker_linear import (  # noqa: E402
    TuckerLinear,
    retract_tucker_modules_,
)
from models.utils import get_model  # noqa: E402
from optim.tensorion import TensorionOptimizer, tucker_core_shape_overrides  # noqa: E402
from scripts.tucker_benchmark.common import (  # noqa: E402
    DEFAULT_SEQUENCE_LENGTH,
    VARIANTS,
    atomic_write_json,
    requested_controls,
    summarize,
    variant_spec,
)


def make_config(args) -> SimpleNamespace:
    static_tucker = args.variant == "static_tucker"
    return SimpleNamespace(
        model="llama",
        vocab_size=50_304,
        sequence_length=args.sequence_length,
        batch_size=args.microbatch,
        n_layer=12,
        n_embd=1024,
        n_head=8,
        dropout=0.0,
        bias=False,
        init_std=0.02,
        rmsnorm_eps=1e-5,
        multiple_of=256,
        ffn_hidden_size=0,
        label_smoothing=0.0,
        qkv_clipping=False,
        qkv_clipping_factor=1.0,
        attention_type="standard",
        linear_parameterization="tucker" if static_tucker else "dense",
        tucker_rank="259",
        tucker_ranks=None,
        tucker_attention_ranks=None,
        tucker_gate_up_ranks=None,
        tucker_down_ranks=None,
        tucker_rank_plan=None,
        tucker_terms=1,
        tucker_equal_params=False,
        tucker_forward_mode="contract",
        tucker_dense_adamw_matrices=static_tucker,
        tucker_retract_every_step=static_tucker,
        tucker_vector_transport=static_tucker,
        tucker_riemannian_muon=static_tucker,
        target_parameter_count=257_676_352,
        target_parameter_tolerance=12_312,
        fp8=False,
        fp8_optim=False,
        qargs=None,
    )


def instantiate_model(config: SimpleNamespace, device: torch.device):
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device(device):
            model = get_model(config)
    finally:
        torch.set_default_dtype(previous_dtype)
    model.to(dtype=torch.bfloat16)
    model.train()
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


def build_tucker_optimizer(args, model):
    groups = parameter_groups(model, args.weight_decay)
    metadata = {}
    for group in groups:
        for parameter in group["params"]:
            metadata[parameter] = (
                float(group["weight_decay"]),
                bool(group.get("is_proj_params", False)),
            )

    factor_parameters = {
        factor
        for module in model.modules()
        if isinstance(module, TuckerLinear)
        for factor in (module.U1, module.U2, module.U3, module.U4)
    }
    logical_shapes = tucker_core_shape_overrides(model)
    tensorion_params = []
    riemannian_muon_params = []
    adamw_by_weight_decay = {}

    for name, parameter in model.named_parameters():
        logical_shape = logical_shapes.get(parameter, tuple(parameter.shape))
        parameter_weight_decay, is_projection = metadata[parameter]
        eligible = is_projection and not any(
            excluded in name
            for excluded in ("wte", "wpe", "lm_head", "embed", "core_logits")
        )
        if eligible and len(logical_shape) >= 3:
            tensorion_params.append((name, parameter, logical_shape))
        elif eligible and len(logical_shape) == 2 and parameter.ndim == 2:
            if parameter not in factor_parameters:
                raise RuntimeError(f"unexpected non-factor matrix in Tucker model: {name}")
            riemannian_muon_params.append((name, parameter))
        else:
            adamw_by_weight_decay.setdefault(parameter_weight_decay, []).append(parameter)

    adamw_groups = [
        {"params": params, "weight_decay": weight_decay}
        for weight_decay, params in adamw_by_weight_decay.items()
    ]
    tucker_modules = [
        (
            name,
            module.core_matrix,
            (module.U1, module.U2, module.U3, module.U4),
        )
        for name, module in model.named_modules()
        if isinstance(module, TuckerLinear)
    ]
    optimizer = TensorionOptimizer(
        tensorion_params=tensorion_params,
        adamw_param_groups=adamw_groups,
        riemannian_muon_params=riemannian_muon_params,
        tucker_module_specs=tucker_modules,
        tucker_lr_scaling_mode="first_order_calibrated",
        tucker_lr_scaling_eps=1e-8,
        tucker_lr_scaling_power_iters=1,
        tucker_lr_scaling_use_stiefel_unit_norm=True,
        tucker_lr_scaling_post_ns_project=False,
        tucker_lr_scaling_stiefel_drift_threshold=1e-3,
        tucker_lr_scaling_strict_bound_check=False,
        tucker_lr_scaling_exact_svd_debug=False,
        tucker_lr_scaling_log_interval=100,
        tucker_riemannian_muon_post_ns_project=False,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
        nesterov=False,
        adjust_lr=True,
        ns_steps=6,
        orthogonalization="ns",
        adamw_betas=(args.beta1, args.beta2),
        adamw_eps=args.eps,
    )
    split = {}
    for group in optimizer.param_groups:
        update_type = group["update_type"]
        split[update_type] = split.get(update_type, 0) + sum(
            parameter.numel() for parameter in group["params"]
        )
    return optimizer, split


def build_model_and_optimizer(args, device: torch.device):
    config = make_config(args)
    model = instantiate_model(config, device)
    if args.variant == "static_tucker":
        forward_modes = {
            module.resolved_forward_mode
            for module in model.modules()
            if isinstance(module, TuckerLinear)
        }
        if forward_modes != {"contract"}:
            raise RuntimeError(
                f"static Tucker benchmark must not materialize weights: {forward_modes}"
            )
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
    else:
        optimizer, split = build_tucker_optimizer(args, model)

        def post_step():
            retract_tucker_modules_(
                model,
                optimizer=optimizer,
                transport_optimizer_state=True,
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


def timed_step(model, optimizer, post_step, batches, stream, grad_clip: float) -> dict:
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

        clip_start.record(stream)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        clip_end.record(stream)
        optimizer_start.record(stream)
        optimizer.step()
        if post_step is not None:
            post_step()
        optimizer_end.record(stream)
        optimizer.zero_grad(set_to_none=True)
        total_end.record(stream)

    torch.cuda.synchronize()
    return {
        "host_total_ms": (time.perf_counter_ns() - host_start) / 1e6,
        "gpu_total_ms": total_start.elapsed_time(total_end),
        "forward_ms": sum(start.elapsed_time(end) for start, end in forward_pairs),
        "backward_ms": sum(start.elapsed_time(end) for start, end in backward_pairs),
        "grad_clip_ms": clip_start.elapsed_time(clip_end),
        "optimizer_ms": optimizer_start.elapsed_time(optimizer_end),
        "loss": sum(float(loss.float()) for loss in losses),
    }


METRICS = (
    "host_total_ms",
    "gpu_total_ms",
    "forward_ms",
    "backward_ms",
    "grad_clip_ms",
    "optimizer_ms",
    "tokens_per_second",
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
    model, optimizer, post_step, optimizer_split = build_model_and_optimizer(args, device)
    actual_parameters = sum(parameter.numel() for parameter in model.parameters())
    if actual_parameters != variant["parameters"]:
        raise RuntimeError(
            f"parameter-count mismatch: actual={actual_parameters}, "
            f"expected={variant['parameters']}"
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

    torch.cuda.reset_peak_memory_stats(device)
    samples = []
    for iteration in range(args.measured_steps):
        sample = timed_step(model, optimizer, post_step, batches, stream, args.grad_clip)
        if not math.isfinite(sample["loss"]):
            raise RuntimeError(f"non-finite measured loss at {iteration}: {sample['loss']}")
        sample["iteration"] = iteration
        sample["tokens_per_second"] = tokens_per_step / (sample["host_total_ms"] / 1000.0)
        samples.append(sample)

    memory = {
        "model_bytes": sum(tensor_bytes(parameter) for parameter in model.parameters()),
        "gradient_bytes_nominal": sum(
            tensor_bytes(parameter) for parameter in model.parameters()
        ),
        "optimizer_state_bytes": tensor_bytes(optimizer.state),
        "optimizer_state_dtypes": sorted(tensor_dtypes(optimizer.state)),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    tucker_stats = getattr(model, "_tucker_replacement_stats", None)
    return {
        "status": "complete",
        "model": {
            "name": "llama-257m",
            "n_layer": 12,
            "n_embd": 1024,
            "n_head": 8,
            "actual_parameters": actual_parameters,
            "tucker_forward_modes": (
                dict(tucker_stats.forward_modes) if tucker_stats is not None else None
            ),
        },
        "variant": variant,
        "benchmark": {
            **requested_controls(args, args.microbatch, args.accumulation_steps),
            "optimizer_backend": (
                "TensorionOptimizer" if args.variant == "static_tucker"
                else "torch.optim.AdamW"
            ),
            "adamw_fused": args.variant == "dense_adamw",
            "cuda_timing": "events on a dedicated stream; device-wide sync at step end",
            "optimizer_parameter_split": optimizer_split,
            "optimizer_ms_includes_tucker_retraction": args.variant == "static_tucker",
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
            metric: summarize([sample[metric] for sample in samples])
            for metric in METRICS
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
    parser.add_argument(
        "--variant",
        required=True,
        choices=[variant["name"] for variant in VARIANTS],
    )
    parser.add_argument("--device", default="cuda:0")
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
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.microbatch, args.sequence_length, args.accumulation_steps) <= 0:
        raise ValueError("microbatch, sequence length and accumulation steps must be positive")
    if args.warmup_steps < 1 or args.measured_steps < 1:
        raise ValueError("warmup and measured steps must be positive")
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
                    "variant": payload["variant"]["name"],
                    "microbatch": args.microbatch,
                    "median_ms": payload["summary"]["host_total_ms"]["median"],
                    "tokens_per_second": payload["summary"]["tokens_per_second"]["median"],
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
