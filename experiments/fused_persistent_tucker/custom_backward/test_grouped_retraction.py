#!/usr/bin/env python3
"""Parity and invariance checks for grouped Tucker retraction."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.grouped_retraction import (
    grouped_retract_tucker_modules_,
)
from models.tucker_linear import TuckerLinear, retract_tucker_modules_


def make_module(in_features, out_features, ranks, mode_layout="balanced4"):
    return TuckerLinear(
        in_features,
        out_features,
        rank=ranks,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        contract_chunk_size=16,
        mode_layout=mode_layout,
        device="cuda",
        dtype=torch.float32,
    )


def main():
    torch.manual_seed(77)
    source = nn.ModuleList(
        [make_module(1024, 1024, (32, 32, 32, 32)) for _ in range(3)]
        + [make_module(1024, 2816, (32, 32, 44, 64)) for _ in range(2)]
        + [make_module(2816, 1024, (44, 64, 32, 32)) for _ in range(2)]
        + [
            make_module(1024, 1024, (16, 16, 64, 1), "order3_input")
            for _ in range(2)
        ]
        + [
            make_module(1024, 1024, (1, 64, 16, 16), "order3_output")
            for _ in range(2)
        ]
    )
    reference = copy.deepcopy(source)
    grouped = copy.deepcopy(source)
    inputs = [
        torch.randn(2, module.in_features, device="cuda") for module in source
    ]
    outputs_before = [module(value) for module, value in zip(source, inputs)]
    retract_tucker_modules_(reference)
    result = grouped_retract_tucker_modules_(grouped, compute_diagnostics=True)
    if result["modules"] != len(grouped):
        raise AssertionError(result)
    expected_factors = sum(len(module.active_factor_names) for module in grouped)
    if result["factors"] != expected_factors:
        raise AssertionError(result)

    for actual, expected in zip(grouped.parameters(), reference.parameters()):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
    outputs_after = [module(value) for module, value in zip(grouped, inputs)]
    for actual, expected in zip(outputs_after, outputs_before):
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
    if result["max_orthogonality_error"] > 2e-5:
        raise AssertionError(result)
    print("PASS grouped retraction parity and orthogonality")


if __name__ == "__main__":
    main()
