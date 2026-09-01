#!/usr/bin/env python3
"""Parity and invariance checks for grouped Tucker retraction."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from types import SimpleNamespace

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
        + [
            make_module(1024, 1024, (56, 120, 120, 1), "order3_paired")
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

    transport_source = nn.ModuleList(
        [
            make_module(1024, 1024, (56, 120, 120, 1), "order3_paired")
            for _ in range(3)
        ]
        + [make_module(1024, 1024, (56, 120, 128, 1), "order3_paired")]
    )
    transport_reference = copy.deepcopy(transport_source)
    transport_grouped = copy.deepcopy(transport_source)
    momentum_by_name = {
        name: torch.randn_like(parameter)
        for name, parameter in transport_source.named_parameters()
    }

    def make_optimizer(model):
        return SimpleNamespace(
            state={
                parameter: {"momentum_buffer": momentum_by_name[name].clone()}
                for name, parameter in model.named_parameters()
            }
        )

    reference_optimizer = make_optimizer(transport_reference)
    grouped_optimizer = make_optimizer(transport_grouped)
    retract_tucker_modules_(
        transport_reference,
        optimizer=reference_optimizer,
        transport_optimizer_state=True,
    )
    transport_result = grouped_retract_tucker_modules_(
        transport_grouped,
        optimizer=grouped_optimizer,
        transport_optimizer_state=True,
        compute_diagnostics=True,
    )
    for (name, actual), (_, expected) in zip(
        transport_grouped.named_parameters(),
        transport_reference.named_parameters(),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(
            grouped_optimizer.state[actual]["momentum_buffer"],
            reference_optimizer.state[expected]["momentum_buffer"],
            rtol=3e-5,
            atol=3e-6,
            msg=lambda message: f"{name}: {message}",
        )
    if transport_result["transported_cores"] != len(transport_grouped):
        raise AssertionError(transport_result)
    if transport_result["transported_factors"] != 3 * len(transport_grouped):
        raise AssertionError(transport_result)
    if transport_result["max_momentum_tangency_error"] > 3e-5:
        raise AssertionError(transport_result)
    print("PASS grouped retraction parity, transport, and orthogonality")


if __name__ == "__main__":
    main()
