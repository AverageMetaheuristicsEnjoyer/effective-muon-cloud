#!/usr/bin/env python3
"""CUDA parity tests for the isolated custom Tucker backward."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parent))

from models.tucker_chunked import chunked_tucker_linear as reference_linear  # noqa: E402
import models.tucker_linear as tucker_linear_module  # noqa: E402
from models.tucker_linear import TuckerLinear  # noqa: E402
from optim.progressive_tucker import expand_tucker_model_to_plan_  # noqa: E402
from experiments.fused_persistent_tucker.custom_backward.ops import (  # noqa: E402
    custom_tucker_linear,
)


def _run(module, x, implementation, chunk_size=16384):
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = implementation(x, module, chunk_size)
        loss = y.float().square().mean()
    loss.backward()
    grads = {
        "x": x.grad.detach().clone(),
        **{
            name: parameter.grad.detach().clone()
            for name, parameter in module.named_parameters()
        },
    }
    return y.detach(), loss.detach(), grads


def _assert_close(name, actual, expected, atol, rtol):
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    if not torch.isfinite(actual).all():
        raise AssertionError(f"{name} contains NaN/Inf")


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(123)
    base = TuckerLinear(
        1024,
        1024,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        contract_chunk_size=16384,
        device="cuda",
    ).train()
    reference = copy.deepcopy(base)
    custom = copy.deepcopy(base)
    x = torch.randn(256, 1024, device="cuda", dtype=torch.bfloat16)

    ref_y, ref_loss, ref_grads = _run(reference, x, reference_linear)
    for policy in ("persistent", "recast", "hybrid_gate_up"):
        test_module = copy.deepcopy(custom)
        implementation = lambda value, module, chunk: custom_tucker_linear(
            value, module, chunk, cache_policy=policy
        )
        y, loss, grads = _run(test_module, x, implementation)
        _assert_close(f"{policy}.forward", y, ref_y, 0.0, 0.0)
        _assert_close(f"{policy}.loss", loss, ref_loss, 1e-6, 1e-6)
        for name, expected in ref_grads.items():
            _assert_close(f"{policy}.{name}", grads[name], expected, 3e-2, 3e-2)

    # Real Llama embeddings stay FP32 under autocast. The custom path must cast
    # both activations and Tucker work tensors to the active autocast dtype.
    fp32_module = copy.deepcopy(base)
    fp32_x = torch.randn(
        64, 1024, device="cuda", dtype=torch.float32
    ).requires_grad_(True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fp32_y = custom_tucker_linear(
            fp32_x, fp32_module, 16384, cache_policy="recast"
        )
        fp32_loss = fp32_y.float().square().mean()
    fp32_loss.backward()
    if fp32_y.dtype != torch.bfloat16 or not torch.isfinite(fp32_x.grad).all():
        raise AssertionError("FP32-input autocast path produced an invalid result")

    # Both asymmetric production shapes must use the same exact VJP.
    for in_features, out_features in ((1024, 2816), (2816, 1024)):
        production = TuckerLinear(
            in_features,
            out_features,
            rank=259,
            bias=False,
            equal_params=False,
            forward_mode="chunked_contract",
            contract_chunk_size=16384,
            device="cuda",
        ).train()
        production_reference = copy.deepcopy(production)
        production_custom = copy.deepcopy(production)
        production_x = torch.randn(
            64, in_features, device="cuda", dtype=torch.bfloat16
        )
        ref_y, ref_loss, ref_grads = _run(
            production_reference, production_x, reference_linear
        )
        implementation = lambda value, module, chunk: custom_tucker_linear(
            value, module, chunk, cache_policy="hybrid_gate_up"
        )
        y, loss, grads = _run(production_custom, production_x, implementation)
        _assert_close(f"{in_features}->{out_features}.forward", y, ref_y, 0.0, 0.0)
        _assert_close(f"{in_features}->{out_features}.loss", loss, ref_loss, 1e-6, 1e-6)
        for name, expected in ref_grads.items():
            _assert_close(
                f"{in_features}->{out_features}.{name}",
                grads[name],
                expected,
                3e-2,
                3e-2,
            )

    # The cross-scale benchmark uses rank-8-aligned factor pairs.  For 2816
    # features that is 32x88, which deliberately exercises the generic path.
    original_factor_pair = tucker_linear_module.balanced_factor_pair
    try:
        tucker_linear_module.balanced_factor_pair = lambda value: (
            (32, 88) if value == 2816 else original_factor_pair(value)
        )
        aligned = TuckerLinear(
            1024,
            2816,
            rank=(32, 32, 32, 64),
            bias=False,
            equal_params=False,
            forward_mode="chunked_contract",
            contract_chunk_size=16384,
            device="cuda",
        ).train()
    finally:
        tucker_linear_module.balanced_factor_pair = original_factor_pair
    aligned_reference = copy.deepcopy(aligned)
    aligned_custom = copy.deepcopy(aligned)
    aligned_x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    ref_y, ref_loss, ref_grads = _run(
        aligned_reference, aligned_x, reference_linear
    )
    implementation = lambda value, module, chunk: custom_tucker_linear(
        value, module, chunk, cache_policy="recast"
    )
    y, loss, grads = _run(aligned_custom, aligned_x, implementation)
    _assert_close("rank8_fallback.forward", y, ref_y, 0.0, 0.0)
    _assert_close("rank8_fallback.loss", loss, ref_loss, 1e-6, 1e-6)
    for name, expected in ref_grads.items():
        _assert_close(f"rank8_fallback.{name}", grads[name], expected, 4e-2, 4e-2)

    # Non-contiguous input and four chunks exercise the generic accumulation.
    wide = torch.randn(256, 2048, device="cuda", dtype=torch.bfloat16)
    noncontiguous = wide[:, ::2]
    if noncontiguous.is_contiguous():
        raise AssertionError("non-contiguous test input unexpectedly contiguous")
    multi_reference = copy.deepcopy(base)
    multi_custom = copy.deepcopy(base)
    ref_y, ref_loss, ref_grads = _run(
        multi_reference, noncontiguous, reference_linear, chunk_size=64
    )
    implementation = lambda value, module, chunk: custom_tucker_linear(
        value, module, chunk, cache_policy="recast"
    )
    y, loss, grads = _run(
        multi_custom, noncontiguous, implementation, chunk_size=64
    )
    _assert_close("multichunk.forward", y, ref_y, 0.0, 0.0)
    _assert_close("multichunk.loss", loss, ref_loss, 1e-6, 1e-6)
    for name, expected in ref_grads.items():
        _assert_close(f"multichunk.{name}", grads[name], expected, 4e-2, 4e-2)

    # Gradient accumulation must add the same second microstep contribution.
    accumulator = copy.deepcopy(base)
    x1 = x.detach().clone().requires_grad_(True)
    x2 = (x * 0.5).detach().clone().requires_grad_(True)
    for value in (x1, x2):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            custom_tucker_linear(
                value, accumulator, 16384, cache_policy="persistent"
            ).float().square().mean().backward()
    for name, parameter in accumulator.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"invalid accumulated gradient for {name}")

    # An optimizer update must invalidate the version-keyed BF16 work cache.
    old_key = accumulator._direct_tucker_work_cache[0]
    optimizer = torch.optim.SGD(accumulator.parameters(), lr=1e-4)
    optimizer.step()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        custom_tucker_linear(x, accumulator, 16384, cache_policy="persistent")
    new_key = accumulator._direct_tucker_work_cache[0]
    if old_key == new_key:
        raise AssertionError("BF16 cache key did not change after optimizer.step()")

    progressive = torch.nn.Module().cuda()
    progressive.layer = TuckerLinear(
        1024,
        2816,
        rank=(29, 30, 25, 36),
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        contract_chunk_size=16384,
        device="cuda",
    ).train()
    progressive_x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16)
    _run(
        progressive.layer,
        progressive_x,
        lambda value, module, chunk: custom_tucker_linear(
            value, module, chunk, cache_policy="recast"
        ),
    )
    progressive.layer.zero_grad(set_to_none=True)
    expand_tucker_model_to_plan_(
        progressive,
        None,
        {"layer": (30, 31, 31, 44)},
        seed=1_001_704,
        verify_function=True,
        verify_rtol=5e-5,
    )
    _, _, progressive_grads = _run(
        progressive.layer,
        progressive_x,
        lambda value, module, chunk: custom_tucker_linear(
            value, module, chunk, cache_policy="recast"
        ),
    )
    for name, parameter in progressive.layer.named_parameters():
        gradient = progressive_grads[name]
        if gradient.shape != parameter.shape or not torch.isfinite(gradient).all():
            raise AssertionError(f"invalid post-growth gradient for {name}")

    print(
        "PASS all production shapes, non-contiguous multi-chunk, accumulation, "
        "finite grads, cache invalidation, progressive growth"
    )


if __name__ == "__main__":
    main()
