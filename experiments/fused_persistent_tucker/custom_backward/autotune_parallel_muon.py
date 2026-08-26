#!/usr/bin/env python3
"""Single-model autotune of Muon core microbatching and CUDA streams."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.grouped_muon import (  # noqa: E402
    GroupedSmallFactorMuonLite,
)
from experiments.fused_persistent_tucker.custom_backward.parallel_muon import (  # noqa: E402
    ParallelGroupedMuonLite,
)
from models.utils import get_model  # noqa: E402
from scripts.benchmark_dense_muon_step import make_config  # noqa: E402
from third_party.lite.muonlite import MuonLite  # noqa: E402


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def named_split(model):
    muon, adamw = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and not any(
            marker in name
            for marker in ("wte", "wpe", "lm_head", "embed", "core_logits")
        ):
            muon.append((name, parameter))
        else:
            adamw.append((name, parameter))
    return muon, adamw


def make_optimizer(kind, muon, adamw, microbatch=4, streams=4):
    cls = {
        "sequential": MuonLite,
        "factor_grouped": GroupedSmallFactorMuonLite,
        "parallel": ParallelGroupedMuonLite,
    }[kind]
    extra = (
        {"core_microbatch": microbatch, "parallel_streams": streams}
        if kind == "parallel"
        else {}
    )
    return cls(
        muon_params=muon,
        adamw_params=adamw,
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
        **extra,
    )


def measure(candidate, muon, adamw, warmup, iterations):
    optimizer = make_optimizer(
        candidate["kind"],
        muon,
        adamw,
        candidate.get("microbatch", 4),
        candidate.get("streams", 4),
    )
    for _ in range(warmup):
        optimizer.step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    values = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        optimizer.step()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end))
    result = {
        **candidate,
        "median_ms": statistics.median(values),
        "p10_ms": percentile(values, 0.1),
        "p90_ms": percentile(values, 0.9),
        "all_ms": values,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "grouped_core_count": getattr(optimizer, "grouped_core_count", 0),
        "grouped_factor_count": getattr(optimizer, "grouped_factor_count", 0),
        "grouped_small_factor_count": getattr(
            optimizer, "grouped_small_factor_count", 0
        ),
    }
    del optimizer
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.manual_seed(19)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    config_args = argparse.Namespace(
        model_type="tucker",
        batch_size=16,
        sequence_length=1024,
        activation_checkpointing=False,
        chunk_size=16384,
        head_chunk_size=2048,
    )
    model = get_model(make_config(config_args)).cuda().train()
    if sum(parameter.numel() for parameter in model.parameters()) != 257676352:
        raise RuntimeError("parameter-count mismatch")
    if not isinstance(model.lm_head, torch.nn.Linear):
        raise RuntimeError("lm_head must remain Dense")
    muon, adamw = named_split(model)
    for _, parameter in muon + adamw:
        parameter.grad = torch.randn_like(parameter)

    candidates = [
        {"kind": "sequential"},
        {"kind": "factor_grouped"},
    ]
    candidates.extend(
        {"kind": "parallel", "microbatch": microbatch, "streams": streams}
        for microbatch in (1, 2, 4, 8)
        for streams in (1, 2, 4)
    )
    results = []
    for candidate in candidates:
        result = measure(candidate, muon, adamw, args.warmup, args.iterations)
        results.append(result)
        print(json.dumps(result), flush=True)
    valid = [result for result in results if result["kind"] == "parallel"]
    selected = min(valid, key=lambda result: result["median_ms"])
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "parameters": 257676352,
        "dense_lm_head": True,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": results,
        "selected": selected,
    }
    print(json.dumps({"selected": selected}))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
