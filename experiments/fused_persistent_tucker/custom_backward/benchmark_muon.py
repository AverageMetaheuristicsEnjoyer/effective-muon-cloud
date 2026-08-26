#!/usr/bin/env python3
"""Cold and steady full-training benchmark for custom Tucker + Muon."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.ops import (  # noqa: E402
    custom_tucker_linear,
)
from models.tucker_linear import retract_tucker_modules_  # noqa: E402
from experiments.fused_persistent_tucker.custom_backward.grouped_retraction import (  # noqa: E402
    grouped_retract_tucker_modules_,
)
from models.utils import get_model  # noqa: E402
from scripts.benchmark_dense_muon_step import make_config, make_muon  # noqa: E402
from experiments.fused_persistent_tucker.custom_backward.grouped_muon import (  # noqa: E402
    GroupedSmallFactorMuonLite,
)
from experiments.fused_persistent_tucker.custom_backward.parallel_muon import (  # noqa: E402
    ParallelGroupedMuonLite,
)


def _mib(value):
    return value / 1024**2


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _tensor_breakdown(named_tensors):
    totals = defaultdict(int)
    for name, tensor in named_tensors:
        totals[f"{str(tensor.dtype).removeprefix('torch.')}"] += (
            tensor.numel() * tensor.element_size()
        )
    return {key: _mib(value) for key, value in sorted(totals.items())}


def _state_tensors(optimizer):
    for parameter, state in optimizer.state.items():
        for key, value in state.items():
            if torch.is_tensor(value):
                yield f"{key}:{tuple(parameter.shape)}", value


def _cache_bytes(model):
    seen = set()
    total = 0
    for module in model.modules():
        cached = getattr(module, "_direct_tucker_work_cache", None)
        if cached is None:
            continue
        for tensor in cached[1]:
            if tensor.data_ptr() not in seen:
                seen.add(tensor.data_ptr())
                total += tensor.numel() * tensor.element_size()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-policy",
        choices=("persistent", "recast", "hybrid_gate_up"),
        default="recast",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--retract", action="store_true")
    parser.add_argument("--grouped-retraction", action="store_true")
    parser.add_argument("--grouped-small-muon", action="store_true")
    parser.add_argument("--parallel-grouped-muon", action="store_true")
    parser.add_argument("--core-microbatch", type=int, default=1)
    parser.add_argument("--muon-streams", type=int, default=2)
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()

    config_args = argparse.Namespace(
        model_type="tucker",
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        activation_checkpointing=False,
        chunk_size=args.chunk_size,
        head_chunk_size=2048,
    )
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = get_model(make_config(config_args)).cuda().train()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    modules = sum(hasattr(module, "materialize_weight") for module in model.modules())
    if parameters != 257676352 or modules != 84:
        raise RuntimeError(f"target mismatch: parameters={parameters}, modules={modules}")
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError("lm_head must remain dense nn.Linear")

    import models.tucker_chunked as dispatch

    dispatch.chunked_tucker_linear = lambda x, module, chunk: custom_tucker_linear(
        x, module, chunk, cache_policy=args.cache_policy
    )
    optimizer_args = argparse.Namespace()
    optimizer, muon_parameters, adamw_parameters = make_muon(model, optimizer_args)
    if args.grouped_small_muon and args.parallel_grouped_muon:
        parser.error("select only one grouped Muon implementation")
    if args.grouped_small_muon or args.parallel_grouped_muon:
        named_muon = []
        named_adamw = []
        for name, parameter in model.named_parameters():
            if parameter.ndim == 2 and not any(
                key in name
                for key in ("wte", "wpe", "lm_head", "embed", "core_logits")
            ):
                named_muon.append((name, parameter))
            else:
                named_adamw.append((name, parameter))
        optimizer_class = (
            ParallelGroupedMuonLite
            if args.parallel_grouped_muon
            else GroupedSmallFactorMuonLite
        )
        parallel_kwargs = (
            {
                "core_microbatch": args.core_microbatch,
                "parallel_streams": args.muon_streams,
            }
            if args.parallel_grouped_muon
            else {}
        )
        optimizer = optimizer_class(
            muon_params=named_muon,
            adamw_params=named_adamw,
            lr=1e-3,
            weight_decay=0.1,
            ns_steps=6,
            muon_theta=0.95,
            adamw_betas=(0.9, 0.99),
            total_steps=39250,
            warmup_steps=2000,
            beta1=0.0,
            beta2=0.0,
            chi=1.0,
            chi_adamw=1.0,
            subspace_ratio=0.0,
            **parallel_kwargs,
        )
    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def timed_step(*, memory=False, validate=False):
        model.zero_grad(set_to_none=True)
        if memory:
            torch.cuda.reset_peak_memory_stats()
        events = [torch.cuda.Event(enable_timing=True) for _ in range(6)]
        events[0].record()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        events[1].record()
        loss.backward()
        events[2].record()
        fb_peak = torch.cuda.max_memory_allocated() if memory else 0
        gradients_present = all(
            parameter.grad is not None for parameter in model.parameters()
        )
        gradients_finite = True
        if validate:
            gradients_finite = all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
        gradient_bytes = sum(
            parameter.grad.numel() * parameter.grad.element_size()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        events[3].record()
        if memory:
            torch.cuda.reset_peak_memory_stats()
        optimizer.step()
        events[4].record()
        optimizer_peak = torch.cuda.max_memory_allocated() if memory else 0
        if memory:
            torch.cuda.reset_peak_memory_stats()
        if args.retract:
            retraction = (
                grouped_retract_tucker_modules_
                if args.grouped_retraction
                else retract_tucker_modules_
            )
            retraction(model)
        events[5].record()
        events[5].synchronize()
        retract_peak = torch.cuda.max_memory_allocated() if memory else 0
        phase_ms = {
            "forward": events[0].elapsed_time(events[1]),
            "backward": events[1].elapsed_time(events[2]),
            "clip": events[2].elapsed_time(events[3]),
            "optimizer": events[3].elapsed_time(events[4]),
            "retract": events[4].elapsed_time(events[5]),
            "full": events[0].elapsed_time(events[5]),
        }
        memory_stats = {
            "peak_fb_mib": _mib(fb_peak),
            "peak_optimizer_mib": _mib(optimizer_peak),
            "peak_retract_mib": _mib(retract_peak),
            "gradient_mib": _mib(gradient_bytes),
        }
        return float(loss), phase_ms, memory_stats, gradients_present, gradients_finite

    model_mib = _mib(
        sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    )
    cold_loss, cold_times, cold_memory, grads_present, grads_finite = timed_step(
        memory=True, validate=True
    )
    for _ in range(max(0, args.warmup - 1)):
        timed_step()

    samples = defaultdict(list)
    losses = []
    steady_memory = None
    for index in range(args.iterations):
        loss, phase_times, memory_stats, _, _ = timed_step(memory=index == 0)
        losses.append(loss)
        for phase, value in phase_times.items():
            samples[phase].append(value)
        if index == 0:
            steady_memory = memory_stats

    state_items = list(_state_tensors(optimizer))
    state_bytes = sum(value.numel() * value.element_size() for _, value in state_items)
    result = {
        "parameters": parameters,
        "tucker_modules": modules,
        "dense_lm_head": True,
        "cache_policy": args.cache_policy,
        "retract": args.retract,
        "grouped_retraction": args.grouped_retraction,
        "grouped_small_muon": args.grouped_small_muon,
        "parallel_grouped_muon": args.parallel_grouped_muon,
        "core_microbatch": args.core_microbatch,
        "muon_streams": args.muon_streams,
        "grouped_small_factor_count": getattr(
            optimizer, "grouped_small_factor_count", 0
        ),
        "grouped_core_count": getattr(optimizer, "grouped_core_count", 0),
        "grouped_factor_count": getattr(optimizer, "grouped_factor_count", 0),
        "muon_parameters": muon_parameters,
        "adamw_parameters": adamw_parameters,
        "model_mib": model_mib,
        "optimizer_state_mib": _mib(state_bytes),
        "optimizer_state_by_dtype_mib": _tensor_breakdown(state_items),
        "bf16_work_cache_mib": _mib(_cache_bytes(model)),
        "cold_loss": cold_loss,
        "cold_times_ms": cold_times,
        "cold_memory": cold_memory,
        "steady_memory": steady_memory,
        "gradients_present": grads_present,
        "gradients_finite": grads_finite,
        "losses": losses,
        "steady": {
            phase: {
                "median_ms": statistics.median(values),
                "p10_ms": _percentile(values, 0.1),
                "p90_ms": _percentile(values, 0.9),
                "all_ms": values,
            }
            for phase, values in samples.items()
        },
    }
    print(json.dumps(result))
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"result={args.output_json}")


if __name__ == "__main__":
    main()
