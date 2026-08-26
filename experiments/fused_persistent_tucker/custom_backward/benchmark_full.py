#!/usr/bin/env python3
"""Interleaved full-model benchmark for custom Tucker backward candidates."""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from experiments.fused_persistent_tucker.tucker_fused_ops import (  # noqa: E402
    clear_work_caches,
    fused_mode_tucker_linear,
    install as install_existing_fused,
)
from models.tucker_chunked import chunked_tucker_linear as reference_linear  # noqa: E402
from models.utils import get_model  # noqa: E402
from experiments.fused_persistent_tucker.custom_backward.ops import (  # noqa: E402
    custom_tucker_linear,
)
from scripts.benchmark_full_tucker_step import make_config  # noqa: E402


VARIANTS = (
    "reference",
    "existing_fused",
    "custom_persistent",
    "custom_hybrid_gate_up",
    "custom_recast",
)


def _percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--chunk-size", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    config_args = argparse.Namespace(
        mode="chunked_contract",
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        layers=12,
        chunk_size=args.chunk_size,
        head_chunk_size=2048,
        activation_checkpointing=False,
    )
    model = get_model(make_config(config_args)).cuda().train()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    module_count = sum(hasattr(module, "materialize_weight") for module in model.modules())
    if parameter_count != 257676352 or module_count != 84:
        raise RuntimeError(
            f"target mismatch: parameters={parameter_count}, modules={module_count}"
        )
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError("lm_head must remain dense nn.Linear")

    import models.tucker_chunked as dispatch

    # Enable the validated analytical backward for the existing-fused control.
    # configure() restores the selected dispatch function before each step.
    install_existing_fused(fused_backward=True, online_ce=False)

    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def configure(name):
        if name == "reference":
            dispatch.chunked_tucker_linear = reference_linear
        elif name == "existing_fused":
            dispatch.chunked_tucker_linear = fused_mode_tucker_linear
        elif name == "custom_persistent":
            dispatch.chunked_tucker_linear = lambda x, module, chunk: custom_tucker_linear(
                x, module, chunk, cache_policy="persistent"
            )
        elif name == "custom_hybrid_gate_up":
            dispatch.chunked_tucker_linear = lambda x, module, chunk: custom_tucker_linear(
                x, module, chunk, cache_policy="hybrid_gate_up"
            )
        elif name == "custom_recast":
            dispatch.chunked_tucker_linear = lambda x, module, chunk: custom_tucker_linear(
                x, module, chunk, cache_policy="recast"
            )
        else:  # pragma: no cover
            raise ValueError(name)

    def step(name):
        configure(name)
        model.zero_grad(set_to_none=True)
        clear_work_caches(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return loss

    for name in VARIANTS:
        for _ in range(args.warmup):
            step(name)
    torch.cuda.synchronize()

    timings = {name: [] for name in VARIANTS}
    peaks_allocated = {name: [] for name in VARIANTS}
    peaks_reserved = {name: [] for name in VARIANTS}
    losses = {}
    for round_index in range(args.rounds):
        shift = round_index % len(VARIANTS)
        order = VARIANTS[shift:] + VARIANTS[:shift]
        for name in order:
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step(name)
            end.record()
            end.synchronize()
            timings[name].append(start.elapsed_time(end))
            peaks_allocated[name].append(torch.cuda.max_memory_allocated() / 1024**2)
            peaks_reserved[name].append(torch.cuda.max_memory_reserved() / 1024**2)
            losses[name] = float(loss)

    print(
        f"parameters={parameter_count} tucker_modules={module_count} "
        f"dense_lm_head={isinstance(model.lm_head, torch.nn.Linear)} "
        f"batch={args.batch_size} sequence={args.sequence_length}"
    )
    for name in VARIANTS:
        values = timings[name]
        print(
            f"variant={name} median_ms={statistics.median(values):.3f} "
            f"p10_ms={_percentile(values, 0.1):.3f} "
            f"p90_ms={_percentile(values, 0.9):.3f} "
            f"all_ms={','.join(f'{value:.3f}' for value in values)} "
            f"peak_allocated_mib={max(peaks_allocated[name]):.1f} "
            f"peak_reserved_mib={max(peaks_reserved[name]):.1f} "
            f"loss={losses[name]:.8f}"
        )


if __name__ == "__main__":
    main()
