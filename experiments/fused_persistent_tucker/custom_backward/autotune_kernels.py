#!/usr/bin/env python3
"""Offline A100 autotuning for the two custom layout-aware VJP kernels."""

from __future__ import annotations

import argparse
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

from experiments.fused_persistent_tucker.custom_backward.kernels import (  # noqa: E402
    _block,
    _input_pair_backward_transposed_kernel,
    _output_pair_backward_transposed_kernel,
)


def bench(launch, warmup=10, iterations=30):
    for _ in range(warmup):
        launch()
    torch.cuda.synchronize()
    values = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        launch()
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end))
    return statistics.median(values), values


def tune_shape(mode, tokens):
    n1, n2, r1, r2 = mode
    device = "cuda"
    dtype = torch.bfloat16
    grad = torch.randn(tokens, r1, r2, device=device, dtype=dtype)
    shaped = torch.randn(tokens, n1, n2, device=device, dtype=dtype)
    U1 = torch.randn(n1, r1, device=device, dtype=dtype)
    U2 = torch.randn(n2, r2, device=device, dtype=dtype)
    grad_first_t = torch.empty(tokens, n2, r1, device=device, dtype=dtype)
    grad_x = torch.empty_like(shaped)
    grouped_input = torch.empty(tokens, n2, n1, device=device, dtype=dtype)

    output_grad = torch.randn(tokens, n1, n2, device=device, dtype=dtype)
    core_out = torch.randn(tokens, r1, r2, device=device, dtype=dtype)
    U3 = torch.randn(n1, r1, device=device, dtype=dtype)
    U4 = torch.randn(n2, r2, device=device, dtype=dtype)
    grad_third_t = torch.empty(tokens, r2, n1, device=device, dtype=dtype)
    grad_core_out = torch.empty_like(core_out)
    grouped_core_out = torch.empty(tokens, r2, r1, device=device, dtype=dtype)

    results = []
    for warps in (4, 8):
        for stages in (2, 3, 4):
            def launch_input(warps=warps, stages=stages):
                _input_pair_backward_transposed_kernel[(tokens,)](
                    grad,
                    shaped,
                    U1,
                    U2,
                    grad_first_t,
                    grad_x,
                    grouped_input,
                    n1=n1,
                    n2=n2,
                    r1=r1,
                    r2=r2,
                    BN1=_block(n1),
                    BN2=_block(n2),
                    BR1=_block(r1),
                    BR2=_block(r2),
                    num_warps=warps,
                    num_stages=stages,
                )

            def launch_output(warps=warps, stages=stages):
                _output_pair_backward_transposed_kernel[(tokens,)](
                    output_grad,
                    core_out,
                    U3,
                    U4,
                    grad_third_t,
                    grad_core_out,
                    grouped_core_out,
                    r3=r1,
                    r4=r2,
                    m1=n1,
                    m2=n2,
                    BR3=_block(r1),
                    BR4=_block(r2),
                    BM1=_block(n1),
                    BM2=_block(n2),
                    num_warps=warps,
                    num_stages=stages,
                )

            input_median, input_values = bench(launch_input)
            output_median, output_values = bench(launch_output)
            results.append(
                {
                    "num_warps": warps,
                    "num_stages": stages,
                    "input_median_ms": input_median,
                    "output_median_ms": output_median,
                    "combined_ms": input_median + output_median,
                    "input_all_ms": input_values,
                    "output_all_ms": output_values,
                }
            )
    results.sort(key=lambda item: item["combined_ms"])
    return {"mode": mode, "results": results, "selected": results[0]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16384)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = {
        "gpu": torch.cuda.get_device_name(),
        "dtype": "bfloat16",
        "tokens": args.tokens,
        "shapes": [
            tune_shape((32, 32, 32, 32), args.tokens),
            tune_shape((44, 64, 44, 64), args.tokens),
        ],
    }
    print(json.dumps(payload))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
