#!/usr/bin/env python3
"""Multi-step parity for stream-parallel core and factor Muon updates."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))

from experiments.fused_persistent_tucker.custom_backward.parallel_muon import (  # noqa: E402
    ParallelGroupedMuonLite,
)
from third_party.lite.muonlite import MuonLite  # noqa: E402


def make_optimizer(cls, named, adamw_named):
    kwargs = dict(
        muon_params=named,
        adamw_params=adamw_named,
        lr=1e-3,
        weight_decay=0.1,
        ns_steps=6,
        muon_theta=0.95,
        beta1=0.0,
        beta2=0.0,
        chi=1.0,
        subspace_ratio=0.0,
    )
    if cls is ParallelGroupedMuonLite:
        kwargs.update(core_microbatch=1, parallel_streams=2)
    return cls(**kwargs)


def main():
    torch.manual_seed(321)
    specs = (
        [("core_matrix", (1024, 1024))] * 2
        + [("core_matrix", (2816, 1024))] * 2
        + [("core_matrix", (1024, 2816))] * 2
        + [("U1", (32, 32))] * 8
        + [("U3", (44, 44))] * 4
        + [("U4", (64, 64))] * 4
    )
    reference_parameters = [
        torch.nn.Parameter(torch.randn(shape, device="cuda")) for _, shape in specs
    ]
    parallel_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in reference_parameters
    ]
    reference_named = [
        (f"layers.{index}.{kind}", parameter)
        for index, ((kind, _), parameter) in enumerate(zip(specs, reference_parameters))
    ]
    parallel_named = [
        (f"layers.{index}.{kind}", parameter)
        for index, ((kind, _), parameter) in enumerate(zip(specs, parallel_parameters))
    ]
    reference_adamw_parameters = [
        torch.nn.Parameter(torch.randn((256, 128), device="cuda")),
        torch.nn.Parameter(torch.randn((128,), device="cuda")),
    ]
    parallel_adamw_parameters = [
        torch.nn.Parameter(parameter.detach().clone())
        for parameter in reference_adamw_parameters
    ]
    reference_adamw_named = [
        ("lm_head.weight", reference_adamw_parameters[0]),
        ("model.norm.weight", reference_adamw_parameters[1]),
    ]
    parallel_adamw_named = [
        ("lm_head.weight", parallel_adamw_parameters[0]),
        ("model.norm.weight", parallel_adamw_parameters[1]),
    ]
    reference = make_optimizer(MuonLite, reference_named, reference_adamw_named)
    parallel = make_optimizer(
        ParallelGroupedMuonLite, parallel_named, parallel_adamw_named
    )
    if parallel.grouped_core_count != 6 or parallel.grouped_factor_count != 16:
        raise AssertionError(
            (parallel.grouped_core_count, parallel.grouped_factor_count)
        )

    for step in range(3):
        torch.manual_seed(700 + step)
        gradients = [torch.randn_like(parameter) for parameter in reference_parameters]
        adamw_gradients = [
            torch.randn_like(parameter) for parameter in reference_adamw_parameters
        ]
        for parameter, gradient in zip(reference_parameters, gradients):
            parameter.grad = gradient.clone()
        for parameter, gradient in zip(parallel_parameters, gradients):
            parameter.grad = gradient.clone()
        for parameter, gradient in zip(reference_adamw_parameters, adamw_gradients):
            parameter.grad = gradient.clone()
        for parameter, gradient in zip(parallel_adamw_parameters, adamw_gradients):
            parameter.grad = gradient.clone()
        reference.step()
        parallel.step()
        torch.cuda.synchronize()
        if step == 0:
            checkpoint = parallel.state_dict()
            if any(
                state.get("use_muon") == 4
                for state in checkpoint["state"].values()
            ):
                raise AssertionError("transient parallel routing leaked to checkpoint")
            parallel.load_state_dict(checkpoint)
            if any(
                parallel.state[parameter].get("use_muon") != 4
                for parameter in parallel_parameters
            ):
                raise AssertionError("parallel routing was not restored after resume")

    for actual, expected in zip(parallel_parameters, reference_parameters):
        torch.testing.assert_close(actual, expected, rtol=7e-4, atol=7e-5)
        torch.testing.assert_close(
            parallel.state[actual]["momentum"],
            reference.state[expected]["momentum"],
            rtol=2e-6,
            atol=1e-7,
        )
    for actual, expected in zip(
        parallel_adamw_parameters, reference_adamw_parameters
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(
            parallel.state[actual]["moment1"],
            reference.state[expected]["moment1"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            parallel.state[actual]["moment2"],
            reference.state[expected]["moment2"],
            rtol=0,
            atol=0,
        )
    print(
        "PASS parallel Muon/AdamW parity and checkpoint resume for three steps"
    )


if __name__ == "__main__":
    main()
