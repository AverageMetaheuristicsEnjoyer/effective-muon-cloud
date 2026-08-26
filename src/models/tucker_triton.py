"""Triton mode-pair contractions used by the chunked Tucker path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development environments
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _input_mode_pair_kernel(
        x_ptr,
        U1_ptr,
        U2_ptr,
        out_ptr,
        n1: tl.constexpr,
        n2: tl.constexpr,
        r1: tl.constexpr,
        r2: tl.constexpr,
        BN1: tl.constexpr,
        BN2: tl.constexpr,
        BR1: tl.constexpr,
        BR2: tl.constexpr,
    ):
        token = tl.program_id(0)
        i = tl.arange(0, BN1)
        j = tl.arange(0, BN2)
        a = tl.arange(0, BR1)
        b = tl.arange(0, BR2)
        x = tl.load(
            x_ptr + token * n1 * n2 + i[:, None] * n2 + j[None, :],
            mask=(i[:, None] < n1) & (j[None, :] < n2),
            other=0.0,
        )
        U1_t = tl.load(
            U1_ptr + i[None, :] * r1 + a[:, None],
            mask=(a[:, None] < r1) & (i[None, :] < n1),
            other=0.0,
        )
        U2 = tl.load(
            U2_ptr + j[:, None] * r2 + b[None, :],
            mask=(j[:, None] < n2) & (b[None, :] < r2),
            other=0.0,
        )
        first = tl.dot(U1_t, x, input_precision="ieee")
        first = first.to(U2.dtype)
        output = tl.dot(first, U2, input_precision="ieee")
        tl.store(
            out_ptr + token * r1 * r2 + a[:, None] * r2 + b[None, :],
            output,
            mask=(a[:, None] < r1) & (b[None, :] < r2),
        )


    @triton.jit
    def _output_mode_pair_kernel(
        x_ptr,
        U3_ptr,
        U4_ptr,
        out_ptr,
        r3: tl.constexpr,
        r4: tl.constexpr,
        m1: tl.constexpr,
        m2: tl.constexpr,
        BR3: tl.constexpr,
        BR4: tl.constexpr,
        BM1: tl.constexpr,
        BM2: tl.constexpr,
    ):
        token = tl.program_id(0)
        c = tl.arange(0, BR3)
        d = tl.arange(0, BR4)
        p = tl.arange(0, BM1)
        q = tl.arange(0, BM2)
        x = tl.load(
            x_ptr + token * r3 * r4 + c[:, None] * r4 + d[None, :],
            mask=(c[:, None] < r3) & (d[None, :] < r4),
            other=0.0,
        )
        U3 = tl.load(
            U3_ptr + p[:, None] * r3 + c[None, :],
            mask=(p[:, None] < m1) & (c[None, :] < r3),
            other=0.0,
        )
        U4_t = tl.load(
            U4_ptr + q[None, :] * r4 + d[:, None],
            mask=(d[:, None] < r4) & (q[None, :] < m2),
            other=0.0,
        )
        third = tl.dot(U3, x, input_precision="ieee")
        third = third.to(U4_t.dtype)
        output = tl.dot(third, U4_t, input_precision="ieee")
        tl.store(
            out_ptr + token * m1 * m2 + p[:, None] * m2 + q[None, :],
            output,
            mask=(p[:, None] < m1) & (q[None, :] < m2),
        )


def _block(value: int) -> int:
    return max(16, triton.next_power_of_2(value))


def can_fuse_input(U1: torch.Tensor, U2: torch.Tensor) -> bool:
    return (
        triton is not None
        and U1.is_cuda
        and max(*U1.shape, *U2.shape) <= 64
    )


def can_fuse_output(U3: torch.Tensor, U4: torch.Tensor) -> bool:
    return (
        triton is not None
        and U3.is_cuda
        and max(*U3.shape, *U4.shape) <= 64
    )


def input_mode_pair(x: torch.Tensor, U1: torch.Tensor, U2: torch.Tensor):
    if not can_fuse_input(U1, U2):
        raise ValueError("Triton input mode-pair kernel supports dimensions <= 64")
    x = x.contiguous()
    U1 = U1.contiguous()
    U2 = U2.contiguous()
    tokens, n1, n2 = x.shape
    r1, r2 = U1.shape[1], U2.shape[1]
    output = torch.empty((tokens, r1, r2), device=x.device, dtype=x.dtype)
    _input_mode_pair_kernel[(tokens,)](
        x,
        U1,
        U2,
        output,
        n1=n1,
        n2=n2,
        r1=r1,
        r2=r2,
        BN1=_block(n1),
        BN2=_block(n2),
        BR1=_block(r1),
        BR2=_block(r2),
        num_warps=8 if max(n1, n2, r1, r2) > 32 else 4,
    )
    return output


def output_mode_pair(x: torch.Tensor, U3: torch.Tensor, U4: torch.Tensor):
    if not can_fuse_output(U3, U4):
        raise ValueError("Triton output mode-pair kernel supports dimensions <= 64")
    x = x.contiguous()
    U3 = U3.contiguous()
    U4 = U4.contiguous()
    tokens, r3, r4 = x.shape
    m1, m2 = U3.shape[0], U4.shape[0]
    output = torch.empty((tokens, m1, m2), device=x.device, dtype=x.dtype)
    _output_mode_pair_kernel[(tokens,)](
        x,
        U3,
        U4,
        output,
        r3=r3,
        r4=r4,
        m1=m1,
        m2=m2,
        BR3=_block(r3),
        BR4=_block(r4),
        BM1=_block(m1),
        BM2=_block(m2),
        num_warps=8 if max(r3, r4, m1, m2) > 32 else 4,
    )
    return output
