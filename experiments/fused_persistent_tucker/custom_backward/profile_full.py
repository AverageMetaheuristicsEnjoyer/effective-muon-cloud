#!/usr/bin/env python3
"""Profile a validated Dense-head Tucker full-model candidate."""

from __future__ import annotations

import argparse
import json
import sys
import time
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
from experiments.fused_persistent_tucker.tucker_fused_ops import (  # noqa: E402
    clear_work_caches,
)
from models.utils import get_model  # noqa: E402
from scripts.benchmark_full_tucker_step import make_config  # noqa: E402


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
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--top", type=int, default=60)
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--summary-json", type=Path)
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
    tokens = torch.randint(
        0, 50304, (args.batch_size, args.sequence_length), device="cuda"
    )
    targets = torch.randint_like(tokens, 0, 50304)

    def step():
        model.zero_grad(set_to_none=True)
        clear_work_caches(model)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(tokens, targets)["loss"]
        loss.backward()
        return loss

    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=(
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - start) * 1000

    averages = profiler.key_averages()
    rows = []
    for item in averages:
        self_device_time = getattr(
            item,
            "self_cuda_time_total",
            getattr(item, "self_device_time_total", 0.0),
        )
        device_time = getattr(
            item,
            "cuda_time_total",
            getattr(item, "device_time_total", 0.0),
        )
        rows.append(
            {
                "key": item.key,
                "count": item.count,
                "self_cuda_ms": self_device_time / 1000.0,
                "cuda_total_ms": device_time / 1000.0,
                "self_cpu_ms": item.self_cpu_time_total / 1000.0,
            }
        )
    rows.sort(key=lambda row: row["self_cuda_ms"], reverse=True)
    payload = {
        "parameters": parameters,
        "tucker_modules": modules,
        "dense_lm_head": True,
        "cache_policy": args.cache_policy,
        "wall_ms": wall_ms,
        "loss": float(loss),
        "events": rows,
    }
    print(json.dumps({key: value for key, value in payload.items() if key != "events"}))
    print(averages.table(sort_by="self_cuda_time_total", row_limit=args.top))
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace))
        print(f"trace={args.trace}")
    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"summary={args.summary_json}")


if __name__ == "__main__":
    main()
