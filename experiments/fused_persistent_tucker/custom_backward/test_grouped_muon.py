#!/usr/bin/env python3
"""CUDA parity test for grouped scheduling of small Tucker factors."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.grouped_muon import (
    GroupedSmallFactorMuonLite,
)
from third_party.lite.muonlite import MuonLite


def make_optimizer(cls, named):
    return cls(
        muon_params=named,
        adamw_params=[],
        lr=1e-3,
        weight_decay=0.1,
        ns_steps=6,
        muon_theta=0.95,
        beta1=0.0,
        beta2=0.0,
        chi=1.0,
        subspace_ratio=0.0,
    )


def main():
    torch.manual_seed(123)
    reference_parameters = [
        torch.nn.Parameter(torch.randn(32, 32, device="cuda")) for _ in range(8)
    ]
    grouped_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in reference_parameters
    ]
    reference = make_optimizer(
        MuonLite,
        [(f"layers.{index}.q_proj.U1", parameter) for index, parameter in enumerate(reference_parameters)],
    )
    grouped = make_optimizer(
        GroupedSmallFactorMuonLite,
        [(f"layers.{index}.q_proj.U1", parameter) for index, parameter in enumerate(grouped_parameters)],
    )
    if grouped.grouped_small_factor_count != len(grouped_parameters):
        raise AssertionError("the synthetic Tucker factors were not grouped")

    for step in range(3):
        torch.manual_seed(900 + step)
        gradients = [torch.randn_like(parameter) for parameter in reference_parameters]
        for parameter, gradient in zip(reference_parameters, gradients):
            parameter.grad = gradient.clone()
        for parameter, gradient in zip(grouped_parameters, gradients):
            parameter.grad = gradient.clone()
        reference.step()
        grouped.step()

    for actual, expected in zip(grouped_parameters, reference_parameters):
        torch.testing.assert_close(actual, expected, rtol=5e-4, atol=5e-5)
    print("PASS grouped small-factor Muon parity for three optimizer steps")


if __name__ == "__main__":
    main()
