"""Autograd integration for the no-output-recompute Tucker backward.

The forward is identical to the validated fused-mode implementation. During
backward, only intermediates needed by the analytical VJP are recomputed. In
particular, the logical layer output is not produced a second time.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.tucker_chunked import _backward_chunk
from models.tucker_triton import can_fuse_input, can_fuse_output
from tucker_fused_ops import (
    _cached_work_tensors,
    _forward_chunk_fast,
    _input_pair_saved,
)

from .kernels import (
    can_use_output_third_only,
    input_pair_backward_transposed,
    output_pair_backward_transposed,
    output_third_only,
)


VALID_CACHE_POLICIES = ("persistent", "recast", "hybrid_gate_up")


def _fresh_work_tensors(module, dtype: torch.dtype):
    with torch.no_grad():
        return tuple(
            parameter.to(dtype=dtype)
            for parameter in (
                module.core_matrix,
                module.U1,
                module.U2,
                module.U3,
                module.U4,
            )
        )


def _recompute_saved_without_output(x, core, U1, U2, U3, U4):
    """Recompute exact VJP inputs, never the discarded Tucker layer output."""
    tokens = x.shape[0]
    shaped = x.reshape(tokens, U1.shape[0], U2.shape[0])
    if can_fuse_input(U1, U2):
        contracted, first = _input_pair_saved(shaped, U1, U2)
    else:
        n1, n2 = U1.shape[0], U2.shape[0]
        r1 = U1.shape[1]
        first = (
            shaped.permute(0, 2, 1).reshape(tokens * n2, n1) @ U1
        ).reshape(tokens, n2, r1).permute(0, 2, 1).contiguous()
        contracted = (first.reshape(tokens * r1, n2) @ U2).reshape(
            tokens, r1, U2.shape[1]
        )

    core_out = F.linear(contracted.flatten(1), core).reshape(
        tokens, U3.shape[1], U4.shape[1]
    )
    if can_fuse_output(U3, U4) and can_use_output_third_only(core_out, U3):
        third = output_third_only(core_out, U3)
    else:
        r3, r4 = U3.shape[1], U4.shape[1]
        m1 = U3.shape[0]
        third = (
            core_out.permute(0, 2, 1).reshape(tokens * r4, r3) @ U3.mT
        ).reshape(tokens, r4, m1).permute(0, 2, 1).contiguous()
    return shaped, first, contracted, core_out, third


def _backward_chunk_transposed(grad_output, saved, core, U1, U2, U3, U4):
    """Analytical VJP whose mode intermediates are born GEMM-ready."""
    shaped, first, contracted, core_out, third = saved
    tokens = shaped.shape[0]
    m1, m2 = U3.shape[0], U4.shape[0]
    r1, r2 = U1.shape[1], U2.shape[1]
    n1, n2 = U1.shape[0], U2.shape[0]
    r3, r4 = U3.shape[1], U4.shape[1]
    grad_output = grad_output.reshape(tokens, m1, m2)

    (
        grad_core_out,
        grouped_grad_third,
        grouped_core_out,
    ) = output_pair_backward_transposed(
        grad_output, core_out, U3, U4
    )
    grad_U4 = grad_output.reshape(tokens * m1, m2).mT @ third.reshape(
        tokens * m1, r4
    )
    grad_U3 = grouped_grad_third.reshape(tokens * r4, m1).mT @ (
        grouped_core_out.reshape(tokens * r4, r3)
    )

    grad_core_flat = grad_core_out.flatten(1)
    grad_core = grad_core_flat.mT @ contracted.flatten(1)
    grad_contracted = (grad_core_flat @ core).reshape(tokens, r1, r2)

    grad_input, grouped_grad_first, grouped_input = input_pair_backward_transposed(
        grad_contracted, shaped, U1, U2
    )
    grad_U2 = first.reshape(tokens * r1, n2).mT @ grad_contracted.reshape(
        tokens * r1, r2
    )
    grad_U1 = grouped_input.reshape(tokens * n2, n1).mT @ (
        grouped_grad_first.reshape(tokens * n2, r1)
    )
    return grad_input.flatten(1), grad_core, grad_U1, grad_U2, grad_U3, grad_U4


class NoOutputRecomputeTuckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        core,
        U1,
        U2,
        U3,
        U4,
        corew,
        U1w,
        U2w,
        U3w,
        U4w,
        chunk_size,
        recast_backward,
    ):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        chunk_size = int(chunk_size)
        outputs = []
        for start in range(0, flat_x.shape[0], chunk_size):
            outputs.append(
                _forward_chunk_fast(
                    flat_x[start : start + chunk_size],
                    corew,
                    U1w,
                    U2w,
                    U3w,
                    U4w,
                )
            )
        flat_output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)
        if bool(recast_backward):
            ctx.save_for_backward(x, core, U1, U2, U3, U4)
        else:
            ctx.save_for_backward(x, corew, U1w, U2w, U3w, U4w)
        ctx.recast_backward = bool(recast_backward)
        ctx.chunk_size = chunk_size
        ctx.original_shape = original_shape
        ctx.output_features = U3.shape[0] * U4.shape[0]
        ctx.parameter_dtypes = tuple(
            parameter.dtype for parameter in (core, U1, U2, U3, U4)
        )
        return flat_output.reshape(*original_shape[:-1], ctx.output_features)

    @staticmethod
    def backward(ctx, grad_output):
        x, core_saved, U1_saved, U2_saved, U3_saved, U4_saved = ctx.saved_tensors
        if ctx.recast_backward:
            with torch.no_grad():
                corew, U1w, U2w, U3w, U4w = tuple(
                    parameter.to(dtype=x.dtype)
                    for parameter in (
                        core_saved,
                        U1_saved,
                        U2_saved,
                        U3_saved,
                        U4_saved,
                    )
                )
        else:
            corew, U1w, U2w, U3w, U4w = (
                core_saved,
                U1_saved,
                U2_saved,
                U3_saved,
                U4_saved,
            )

        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad_output = grad_output.reshape(-1, ctx.output_features)
        # Production B16xS1024 uses one 16,384-token chunk. In that case the
        # analytical VJP already owns exactly the tensors autograd needs, so do
        # not allocate/copy a second dX or zero/add a second parameter-gradient
        # buffer. Multi-chunk inputs retain the stable FP32 accumulation path.
        if flat_x.shape[0] <= ctx.chunk_size:
            saved = _recompute_saved_without_output(
                flat_x, corew, U1w, U2w, U3w, U4w
            )
            backward_chunk = (
                _backward_chunk_transposed
                if can_fuse_input(U1w, U2w) and can_fuse_output(U3w, U4w)
                else _backward_chunk
            )
            grads = backward_chunk(
                flat_grad_output.to(dtype=flat_x.dtype),
                saved,
                corew,
                U1w,
                U2w,
                U3w,
                U4w,
            )
            parameter_grads = tuple(
                gradient.to(dtype=dtype)
                for gradient, dtype in zip(grads[1:], ctx.parameter_dtypes)
            )
            return (
                grads[0].reshape(ctx.original_shape),
                *parameter_grads,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )

        grad_x = torch.empty_like(flat_x)
        accumulators = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in (corew, U1w, U2w, U3w, U4w)
        ]
        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_x.shape[0])
            saved = _recompute_saved_without_output(
                flat_x[start:stop], corew, U1w, U2w, U3w, U4w
            )
            backward_chunk = (
                _backward_chunk_transposed
                if can_fuse_input(U1w, U2w) and can_fuse_output(U3w, U4w)
                else _backward_chunk
            )
            grads = backward_chunk(
                flat_grad_output[start:stop].to(dtype=flat_x.dtype),
                saved,
                corew,
                U1w,
                U2w,
                U3w,
                U4w,
            )
            grad_x[start:stop].copy_(grads[0].to(dtype=grad_x.dtype))
            for accumulator, value in zip(accumulators, grads[1:]):
                accumulator.add_(value.float())
        parameter_grads = tuple(
            gradient.to(dtype=dtype)
            for gradient, dtype in zip(accumulators, ctx.parameter_dtypes)
        )
        return (
            grad_x.reshape(ctx.original_shape),
            *parameter_grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def custom_tucker_linear(x, module, chunk_size: int, *, cache_policy: str):
    if cache_policy not in VALID_CACHE_POLICIES:
        raise ValueError(
            f"cache_policy must be one of {VALID_CACHE_POLICIES}, got {cache_policy!r}"
        )
    # The hybrid policy spends memory only on the 24 largest 1024->2816
    # Gate/Up cores. Their BF16 copies cost about 135 MiB but cover roughly 45%
    # of all internal core elements, keeping the measured full-step peak below
    # the Dense reference while reducing backward recast traffic.
    recast = cache_policy == "recast" or (
        cache_policy == "hybrid_gate_up" and module.out_features != 2816
    )
    work_dtype = (
        torch.get_autocast_dtype(x.device.type)
        if torch.is_autocast_enabled(x.device.type)
        else x.dtype
    )
    work = (
        _fresh_work_tensors(module, work_dtype)
        if recast
        else _cached_work_tensors(module, work_dtype)
    )
    parameters = tuple(
        parameter.view_as(parameter)
        for parameter in (
            module.core_matrix,
            module.U1,
            module.U2,
            module.U3,
            module.U4,
        )
    )
    return NoOutputRecomputeTuckerLinearFunction.apply(
        x.to(dtype=work_dtype),
        *parameters,
        *work,
        chunk_size,
        recast,
    )
