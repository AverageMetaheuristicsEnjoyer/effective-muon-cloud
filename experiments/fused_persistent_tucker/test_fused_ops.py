#!/usr/bin/env python3
"""CUDA parity tests for the experimental fused direct Tucker operators."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


HERE = Path(__file__).resolve()
ROOT = next(
    candidate
    for candidate in (HERE.parents[1], HERE.parents[2])
    if (candidate / "src" / "models").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from models.tucker_chunked import (  # noqa: E402
    chunked_tucker_cross_entropy as reference_ce,
    chunked_tucker_linear as reference_linear,
)
from models.tucker_linear import TuckerLinear  # noqa: E402
from tucker_fused_ops import (  # noqa: E402
    clear_work_caches,
    fused_mode_tucker_cross_entropy,
    fused_mode_tucker_linear,
)
from tucker_online_ce import online_tucker_cross_entropy  # noqa: E402


def _clone_grads(x, module):
    return [x.grad.detach().clone()] + [
        parameter.grad.detach().clone() for parameter in module.parameters()
    ]


def _zero_grads(x, module):
    x.grad = None
    for parameter in module.parameters():
        parameter.grad = None


def _assert_grad_parity(left, right):
    for index, (actual, expected) in enumerate(zip(left, right)):
        torch.testing.assert_close(
            actual, expected, rtol=3e-2, atol=3e-2,
            msg=lambda msg: f"gradient {index}: {msg}",
        )


def test_linear():
    torch.manual_seed(11)
    module = TuckerLinear(
        64,
        96,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        device="cuda",
    )
    x = torch.randn(
        2, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    grad_output = torch.randn(2, 64, 96, device="cuda", dtype=torch.bfloat16)
    reference = reference_linear(x, module, 128)
    (reference * grad_output).sum().backward()
    reference_grads = _clone_grads(x, module)
    _zero_grads(x, module)
    actual = fused_mode_tucker_linear(x, module, 128)
    (actual * grad_output).sum().backward()
    actual_grads = _clone_grads(x, module)
    torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
    _assert_grad_parity(actual_grads, reference_grads)


def test_ce():
    torch.manual_seed(12)
    module = TuckerLinear(
        64,
        128,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        device="cuda",
    )
    x = torch.randn(
        2, 64, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randint(0, 128, (2, 64), device="cuda")
    targets[0, 0] = -1
    reference = reference_ce(x, targets, module, 64)
    reference.backward()
    reference_grads = _clone_grads(x, module)
    _zero_grads(x, module)
    actual = fused_mode_tucker_cross_entropy(x, targets, module, 64)
    actual.backward()
    actual_grads = _clone_grads(x, module)
    torch.testing.assert_close(actual, reference, rtol=3e-2, atol=3e-2)
    _assert_grad_parity(actual_grads, reference_grads)

    # Updating a parameter must invalidate the persistent work cache.
    before = module._direct_tucker_work_cache[0]
    with torch.no_grad():
        module.core_matrix.add_(1e-4)
    _zero_grads(x, module)
    fused_mode_tucker_cross_entropy(x, targets, module, 64).backward()
    after = module._direct_tucker_work_cache[0]
    assert before != after
    clear_work_caches(module)
    assert not hasattr(module, "_direct_tucker_work_cache")


def test_online_ce():
    torch.manual_seed(13)
    module = TuckerLinear(
        64,
        128,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        device="cuda",
    )
    x = torch.randn(
        3, 37, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    targets = torch.randint(0, 128, (3, 37), device="cuda")
    targets[0, 0] = -1
    reference = reference_ce(
        x, targets, module, 29, ignore_index=-1, label_smoothing=0.1
    )
    reference.backward()
    reference_grads = _clone_grads(x, module)
    _zero_grads(x, module)
    work = tuple(
        parameter.to(dtype=x.dtype)
        for parameter in (
            module.core_matrix,
            module.U1,
            module.U2,
            module.U3,
            module.U4,
        )
    )
    actual = online_tucker_cross_entropy(
        x,
        targets,
        module,
        work,
        token_chunk_size=29,
        output_mode_tile=3,
        ignore_index=-1,
        label_smoothing=0.1,
    )
    actual.backward()
    actual_grads = _clone_grads(x, module)
    torch.testing.assert_close(actual, reference, rtol=2e-2, atol=2e-2)
    _assert_grad_parity(actual_grads, reference_grads)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    test_linear()
    test_ce()
    test_online_ce()
    print("fused Tucker parity: PASS")
