"""Triton kernels used only by the custom-backward autoresearch path."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU development host
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _output_third_only_kernel(
        core_out_ptr,
        U3_ptr,
        third_ptr,
        r3: tl.constexpr,
        r4: tl.constexpr,
        m1: tl.constexpr,
        BR3: tl.constexpr,
        BR4: tl.constexpr,
        BM1: tl.constexpr,
    ):
        """Compute U3 @ core_out but deliberately skip the unused U4 output."""
        token = tl.program_id(0)
        c = tl.arange(0, BR3)
        d = tl.arange(0, BR4)
        p = tl.arange(0, BM1)
        core_out = tl.load(
            core_out_ptr + token * r3 * r4 + c[:, None] * r4 + d[None, :],
            mask=(c[:, None] < r3) & (d[None, :] < r4),
            other=0.0,
        )
        U3 = tl.load(
            U3_ptr + p[:, None] * r3 + c[None, :],
            mask=(p[:, None] < m1) & (c[None, :] < r3),
            other=0.0,
        )
        third = tl.dot(U3, core_out, input_precision="ieee")
        tl.store(
            third_ptr + token * m1 * r4 + p[:, None] * r4 + d[None, :],
            third,
            mask=(p[:, None] < m1) & (d[None, :] < r4),
        )


    @triton.jit
    def _output_pair_backward_transposed_kernel(
        grad_ptr,
        core_out_ptr,
        U3_ptr,
        U4_ptr,
        grad_third_t_ptr,
        grad_core_out_ptr,
        grouped_core_out_ptr,
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
        grad = tl.load(
            grad_ptr + token * m1 * m2 + p[:, None] * m2 + q[None, :],
            mask=(p[:, None] < m1) & (q[None, :] < m2),
            other=0.0,
        )
        U4 = tl.load(
            U4_ptr + q[:, None] * r4 + d[None, :],
            mask=(q[:, None] < m2) & (d[None, :] < r4),
            other=0.0,
        )
        grad_third = tl.dot(grad, U4, input_precision="ieee")
        # dU3 consumes [token, r4, m1]. Store that layout directly.
        tl.store(
            grad_third_t_ptr + token * r4 * m1 + d[:, None] * m1 + p[None, :],
            tl.trans(grad_third),
            mask=(d[:, None] < r4) & (p[None, :] < m1),
        )
        U3_t = tl.load(
            U3_ptr + p[None, :] * r3 + c[:, None],
            mask=(c[:, None] < r3) & (p[None, :] < m1),
            other=0.0,
        )
        grad_core_out = tl.dot(
            U3_t, grad_third.to(U3_t.dtype), input_precision="ieee"
        )
        tl.store(
            grad_core_out_ptr
            + token * r3 * r4
            + c[:, None] * r4
            + d[None, :],
            grad_core_out,
            mask=(c[:, None] < r3) & (d[None, :] < r4),
        )
        # dU3 also needs core_out as [token, r4, r3]. Fold that layout write
        # into this already-scheduled per-token kernel.
        core_out = tl.load(
            core_out_ptr + token * r3 * r4 + c[:, None] * r4 + d[None, :],
            mask=(c[:, None] < r3) & (d[None, :] < r4),
            other=0.0,
        )
        tl.store(
            grouped_core_out_ptr
            + token * r4 * r3
            + d[:, None] * r3
            + c[None, :],
            tl.trans(core_out),
            mask=(d[:, None] < r4) & (c[None, :] < r3),
        )


    @triton.jit
    def _input_pair_backward_transposed_kernel(
        grad_ptr,
        x_ptr,
        U1_ptr,
        U2_ptr,
        grad_first_t_ptr,
        grad_x_ptr,
        grouped_input_ptr,
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
        grad = tl.load(
            grad_ptr + token * r1 * r2 + a[:, None] * r2 + b[None, :],
            mask=(a[:, None] < r1) & (b[None, :] < r2),
            other=0.0,
        )
        U2_t = tl.load(
            U2_ptr + j[None, :] * r2 + b[:, None],
            mask=(b[:, None] < r2) & (j[None, :] < n2),
            other=0.0,
        )
        grad_first = tl.dot(grad, U2_t, input_precision="ieee")
        # dU1 consumes [token, n2, r1]. Store that layout directly.
        tl.store(
            grad_first_t_ptr + token * n2 * r1 + j[:, None] * r1 + a[None, :],
            tl.trans(grad_first),
            mask=(j[:, None] < n2) & (a[None, :] < r1),
        )
        U1 = tl.load(
            U1_ptr + i[:, None] * r1 + a[None, :],
            mask=(i[:, None] < n1) & (a[None, :] < r1),
            other=0.0,
        )
        grad_x = tl.dot(
            U1, grad_first.to(U1.dtype), input_precision="ieee"
        )
        tl.store(
            grad_x_ptr + token * n1 * n2 + i[:, None] * n2 + j[None, :],
            grad_x,
            mask=(i[:, None] < n1) & (j[None, :] < n2),
        )
        # dU1 consumes x as [token, n2, n1]. Store it while this program owns
        # the same token tile, avoiding a separate permute/copy kernel.
        x = tl.load(
            x_ptr + token * n1 * n2 + i[:, None] * n2 + j[None, :],
            mask=(i[:, None] < n1) & (j[None, :] < n2),
            other=0.0,
        )
        tl.store(
            grouped_input_ptr
            + token * n2 * n1
            + j[:, None] * n1
            + i[None, :],
            tl.trans(x),
            mask=(j[:, None] < n2) & (i[None, :] < n1),
        )


def _block(value: int) -> int:
    if triton is None:
        raise RuntimeError("Triton is required for the custom Tucker kernel")
    return max(16, triton.next_power_of_2(value))


def can_use_output_third_only(core_out: torch.Tensor, U3: torch.Tensor) -> bool:
    return (
        triton is not None
        and core_out.is_cuda
        and core_out.is_contiguous()
        and U3.is_contiguous()
        and max(core_out.shape[1], core_out.shape[2], *U3.shape) <= 64
    )


def output_third_only(core_out: torch.Tensor, U3: torch.Tensor) -> torch.Tensor:
    """Return ``U3 @ core_out`` for each token without constructing output."""
    if not can_use_output_third_only(core_out, U3):
        raise ValueError("output_third_only supports contiguous CUDA modes <= 64")
    tokens, r3, r4 = core_out.shape
    m1 = U3.shape[0]
    third = torch.empty(
        (tokens, m1, r4), device=core_out.device, dtype=core_out.dtype
    )
    _output_third_only_kernel[(tokens,)](
        core_out,
        U3,
        third,
        r3=r3,
        r4=r4,
        m1=m1,
        BR3=_block(r3),
        BR4=_block(r4),
        BM1=_block(m1),
        num_warps=8 if max(r3, r4, m1) > 32 else 4,
    )
    return third


def output_pair_backward_transposed(grad, core_out, U3, U4):
    tokens, m1, m2 = grad.shape
    r3, r4 = U3.shape[1], U4.shape[1]
    grad_third_t = torch.empty(
        (tokens, r4, m1), device=grad.device, dtype=grad.dtype
    )
    grad_core_out = torch.empty(
        (tokens, r3, r4), device=grad.device, dtype=grad.dtype
    )
    grouped_core_out = torch.empty(
        (tokens, r4, r3), device=grad.device, dtype=grad.dtype
    )
    _output_pair_backward_transposed_kernel[(tokens,)](
        grad,
        core_out,
        U3,
        U4,
        grad_third_t,
        grad_core_out,
        grouped_core_out,
        r3=r3,
        r4=r4,
        m1=m1,
        m2=m2,
        BR3=_block(r3),
        BR4=_block(r4),
        BM1=_block(m1),
        BM2=_block(m2),
        # A100 offline autotune at the production 16,384-token chunk selected
        # 4 warps / 4 stages for both 32^4 and 44x64 mode families.
        num_warps=4,
        num_stages=4,
    )
    return grad_core_out, grad_third_t, grouped_core_out


def input_pair_backward_transposed(grad, shaped_input, U1, U2):
    tokens, r1, r2 = grad.shape
    n1, n2 = U1.shape[0], U2.shape[0]
    grad_first_t = torch.empty(
        (tokens, n2, r1), device=grad.device, dtype=grad.dtype
    )
    grad_x = torch.empty(
        (tokens, n1, n2), device=grad.device, dtype=grad.dtype
    )
    grouped_input = torch.empty(
        (tokens, n2, n1), device=grad.device, dtype=grad.dtype
    )
    _input_pair_backward_transposed_kernel[(tokens,)](
        grad,
        shaped_input,
        U1,
        U2,
        grad_first_t,
        grad_x,
        grouped_input,
        n1=n1,
        n2=n2,
        r1=r1,
        r2=r2,
        BN1=_block(n1),
        BN2=_block(n2),
        BR1=_block(r1),
        BR2=_block(r2),
        num_warps=4,
        num_stages=4,
    )
    return grad_x, grad_first_t, grouped_input
