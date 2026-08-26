#!/usr/bin/env python3
"""Parity test using the production 1024 -> 50304 Tucker head shape."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve()
ROOT = next(
    candidate
    for candidate in (HERE.parents[1], HERE.parents[2])
    if (candidate / "src" / "models").is_dir()
)
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE.parent))

from models.tucker_chunked import (  # noqa: E402
    _forward_chunk,
    chunked_tucker_cross_entropy,
)
from models.tucker_linear import TuckerLinear  # noqa: E402
from tucker_online_ce import (  # noqa: E402
    _input_forward,
    _projected_core_tile,
    online_tucker_cross_entropy,
)


def main():
    torch.manual_seed(21)
    module = TuckerLinear(
        1024,
        50304,
        rank=259,
        bias=False,
        equal_params=False,
        forward_mode="chunked_contract",
        device="cuda",
    )
    x = torch.randn(7, 1024, device="cuda", dtype=torch.bfloat16)
    x.requires_grad_(True)
    targets = torch.randint(0, 50304, (7,), device="cuda")
    targets[0] = -1

    reference_loss = chunked_tucker_cross_entropy(
        x,
        targets,
        module,
        7,
        ignore_index=-1,
        label_smoothing=0.1,
    )
    reference_loss.backward()
    reference_grads = [x.grad.detach().clone()] + [
        parameter.grad.detach().clone() for parameter in module.parameters()
    ]
    x.grad = None
    for parameter in module.parameters():
        parameter.grad = None

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
    contracted, _, _ = _input_forward(x, work[1], work[2], save_first=False)
    projected_tiles = []
    for p_start in range(0, work[3].shape[0], 32):
        weight_tile, _, _ = _projected_core_tile(
            work[0], work[3][p_start : p_start + 32], work[4]
        )
        projected_tiles.append(contracted.flatten(1) @ weight_tile.mT)
    projected_logits = torch.cat(projected_tiles, dim=1)
    reference_logits, _ = _forward_chunk(x, *work)
    explicit_loss = F.cross_entropy(
        reference_logits.float(),
        targets,
        ignore_index=-1,
        label_smoothing=0.1,
    )
    projected_loss = F.cross_entropy(
        projected_logits.float(),
        targets,
        ignore_index=-1,
        label_smoothing=0.1,
    )
    online_loss = online_tucker_cross_entropy(
        x,
        targets,
        module,
        work,
        token_chunk_size=7,
        output_mode_tile=32,
        ignore_index=-1,
        label_smoothing=0.1,
    )
    online_loss.backward()
    actual_grads = [x.grad] + [parameter.grad for parameter in module.parameters()]

    torch.testing.assert_close(online_loss, reference_loss, rtol=2e-2, atol=2e-2)
    names = ("x", "core", "U1", "U2", "U3", "U4")
    for name, actual, expected in zip(names, actual_grads, reference_grads):
        torch.testing.assert_close(actual, expected, rtol=8e-2, atol=5e-2)
        print(
            f"{name}: max_abs={(actual.float()-expected.float()).abs().max().item():.6g}"
        )
    print(
        f"production head online CE parity: PASS "
        f"reference_loss={float(reference_loss):.8f} "
        f"explicit_loss={float(explicit_loss):.8f} "
        f"projected_loss={float(projected_loss):.8f} "
        f"online_loss={float(online_loss):.8f} "
        f"logits_max_abs={(projected_logits.float()-reference_logits.float()).abs().max().item():.6g}"
    )


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    main()
