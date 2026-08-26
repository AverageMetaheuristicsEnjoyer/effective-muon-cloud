#!/usr/bin/env python3
"""A100 microbenchmark for grouped QKV and Gate/Up Tucker core GEMMs."""

from __future__ import annotations

import statistics

import torch


def measure(label, function, warmup=10, iterations=50):
    for _ in range(warmup):
        output = function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        output = function()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    print(f"{label}: median_ms={statistics.median(samples):.6f}")
    return output


def main():
    torch.manual_seed(17)
    tokens = 16 * 1024
    dtype = torch.bfloat16

    qkv_x = torch.randn(3, tokens, 1024, device="cuda", dtype=dtype)
    qkv_w = torch.randn(3, 1024, 1024, device="cuda", dtype=dtype)
    separate = measure(
        "qkv_separate_core_mm",
        lambda: tuple(qkv_x[index] @ qkv_w[index].mT for index in range(3)),
    )
    grouped = measure("qkv_grouped_core_bmm", lambda: torch.bmm(qkv_x, qkv_w.mT))
    for index in range(3):
        torch.testing.assert_close(grouped[index], separate[index])

    gate_x = torch.randn(2, tokens, 1024, device="cuda", dtype=dtype)
    gate_w = torch.randn(2, 2816, 1024, device="cuda", dtype=dtype)
    separate = measure(
        "gate_up_separate_core_mm",
        lambda: tuple(gate_x[index] @ gate_w[index].mT for index in range(2)),
    )
    grouped = measure(
        "gate_up_grouped_core_bmm", lambda: torch.bmm(gate_x, gate_w.mT)
    )
    for index in range(2):
        torch.testing.assert_close(grouped[index], separate[index])


if __name__ == "__main__":
    main()
