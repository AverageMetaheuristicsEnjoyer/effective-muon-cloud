#!/usr/bin/env python3
"""Measure grouped core GEMMs for QKV and Gate/Up, including VJPs."""

from __future__ import annotations

import json
import statistics

import torch


def measure(function, warmup=10, iterations=50):
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
    return output, statistics.median(samples), samples


def one_group(name, groups, tokens, in_features, out_features):
    dtype = torch.bfloat16
    x = torch.randn(groups, tokens, in_features, device="cuda", dtype=dtype)
    weight = torch.randn(
        groups, out_features, in_features, device="cuda", dtype=dtype
    )
    grad = torch.randn(groups, tokens, out_features, device="cuda", dtype=dtype)

    functions = {
        "forward_separate": lambda: tuple(
            x[index] @ weight[index].mT for index in range(groups)
        ),
        "forward_grouped": lambda: torch.bmm(x, weight.mT),
        "dx_separate": lambda: tuple(
            grad[index] @ weight[index] for index in range(groups)
        ),
        "dx_grouped": lambda: torch.bmm(grad, weight),
        "dw_separate": lambda: tuple(
            grad[index].mT @ x[index] for index in range(groups)
        ),
        "dw_grouped": lambda: torch.bmm(grad.mT, x),
    }
    outputs = {}
    results = {}
    for key, function in functions.items():
        output, median, samples = measure(function)
        outputs[key] = output
        results[key] = {"median_ms": median, "all_ms": samples}
    for phase in ("forward", "dx", "dw"):
        separate = outputs[f"{phase}_separate"]
        grouped = outputs[f"{phase}_grouped"]
        max_abs = 0.0
        relative_l2 = 0.0
        for index in range(groups):
            difference = (grouped[index].float() - separate[index].float())
            max_abs = max(max_abs, float(difference.abs().max()))
            relative_l2 = max(
                relative_l2,
                float(
                    torch.linalg.vector_norm(difference)
                    / torch.linalg.vector_norm(separate[index].float()).clamp_min(1e-12)
                ),
            )
        if relative_l2 > 5e-3:
            raise AssertionError(
                f"{name} {phase} grouped GEMM relative L2 error "
                f"{relative_l2:.6g} is too large"
            )
        results[phase] = {
            "saved_ms": (
                results[f"{phase}_separate"]["median_ms"]
                - results[f"{phase}_grouped"]["median_ms"]
            ),
            "max_abs_error": max_abs,
            "relative_l2_error": relative_l2,
        }
    return {"name": name, "groups": groups, "results": results}


def main():
    torch.manual_seed(17)
    payload = {
        "qkv": one_group("qkv", 3, 16 * 1024, 1024, 1024),
        "gate_up": one_group("gate_up", 2, 16 * 1024, 1024, 2816),
    }
    print(json.dumps(payload))


if __name__ == "__main__":
    main()
