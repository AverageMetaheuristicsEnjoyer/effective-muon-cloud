"""Experimental direct Tucker operators.

The implementation keeps the logical weight factorised at all times.  It adds
two optimisations on top of ``models.tucker_chunked``:

* mode-pair Triton kernels are used for full 16k-token chunks, not only tiny
  chunks;
* BF16 work copies of FP32 Tucker parameters persist for a parameter version
  and are saved from forward to backward, avoiding a second full recast.

The cache is automatically invalidated by PyTorch's parameter ``_version``
counter after an in-place optimizer update.
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

from models.tucker_chunked import (
    _backward_chunk,
    _forward_chunk,
    _fused_ce_sum_and_grad,
)
from models.tucker_triton import (
    can_fuse_input,
    can_fuse_output,
    input_mode_pair,
    output_mode_pair,
)


_ENABLE_UNVALIDATED_FUSED_BACKWARD = (
    os.environ.get("TUCKER_EXPERIMENTAL_FUSED_BACKWARD", "0") == "1"
)
_ENABLE_ONLINE_CE = os.environ.get("TUCKER_EXPERIMENTAL_ONLINE_CE", "0") == "1"
_ONLINE_CE_OUTPUT_TILE = int(os.environ.get("TUCKER_ONLINE_CE_OUTPUT_TILE", "32"))


if triton is not None:

    @triton.jit
    def _subtract_ce_target_kernel(
        grad_ptr,
        target_ptr,
        columns: tl.constexpr,
        ignore_index: tl.constexpr,
        scale: tl.constexpr,
    ):
        row = tl.program_id(0)
        target = tl.load(target_ptr + row)
        valid = target != ignore_index
        offset = row * columns + target
        value = tl.load(grad_ptr + offset, mask=valid, other=0.0)
        tl.store(grad_ptr + offset, value - scale, mask=valid)


    @triton.jit
    def _input_pair_saved_kernel(
        x_ptr, U1_ptr, U2_ptr, first_ptr, out_ptr,
        n1: tl.constexpr, n2: tl.constexpr,
        r1: tl.constexpr, r2: tl.constexpr,
        BN1: tl.constexpr, BN2: tl.constexpr,
        BR1: tl.constexpr, BR2: tl.constexpr,
    ):
        token = tl.program_id(0)
        i, j = tl.arange(0, BN1), tl.arange(0, BN2)
        a, b = tl.arange(0, BR1), tl.arange(0, BR2)
        x = tl.load(
            x_ptr + token * n1 * n2 + i[:, None] * n2 + j[None, :],
            mask=(i[:, None] < n1) & (j[None, :] < n2), other=0.0,
        )
        U1_t = tl.load(
            U1_ptr + i[None, :] * r1 + a[:, None],
            mask=(a[:, None] < r1) & (i[None, :] < n1), other=0.0,
        )
        U2 = tl.load(
            U2_ptr + j[:, None] * r2 + b[None, :],
            mask=(j[:, None] < n2) & (b[None, :] < r2), other=0.0,
        )
        first = tl.dot(U1_t, x, input_precision="ieee")
        tl.store(
            first_ptr + token * r1 * n2 + a[:, None] * n2 + j[None, :],
            first, mask=(a[:, None] < r1) & (j[None, :] < n2),
        )
        output = tl.dot(first.to(U2.dtype), U2, input_precision="ieee")
        tl.store(
            out_ptr + token * r1 * r2 + a[:, None] * r2 + b[None, :],
            output, mask=(a[:, None] < r1) & (b[None, :] < r2),
        )


    @triton.jit
    def _output_pair_saved_kernel(
        x_ptr, U3_ptr, U4_ptr, third_ptr, out_ptr,
        r3: tl.constexpr, r4: tl.constexpr,
        m1: tl.constexpr, m2: tl.constexpr,
        BR3: tl.constexpr, BR4: tl.constexpr,
        BM1: tl.constexpr, BM2: tl.constexpr,
    ):
        token = tl.program_id(0)
        c, d = tl.arange(0, BR3), tl.arange(0, BR4)
        p, q = tl.arange(0, BM1), tl.arange(0, BM2)
        x = tl.load(
            x_ptr + token * r3 * r4 + c[:, None] * r4 + d[None, :],
            mask=(c[:, None] < r3) & (d[None, :] < r4), other=0.0,
        )
        U3 = tl.load(
            U3_ptr + p[:, None] * r3 + c[None, :],
            mask=(p[:, None] < m1) & (c[None, :] < r3), other=0.0,
        )
        U4_t = tl.load(
            U4_ptr + q[None, :] * r4 + d[:, None],
            mask=(d[:, None] < r4) & (q[None, :] < m2), other=0.0,
        )
        third = tl.dot(U3, x, input_precision="ieee")
        tl.store(
            third_ptr + token * m1 * r4 + p[:, None] * r4 + d[None, :],
            third, mask=(p[:, None] < m1) & (d[None, :] < r4),
        )
        output = tl.dot(third.to(U4_t.dtype), U4_t, input_precision="ieee")
        tl.store(
            out_ptr + token * m1 * m2 + p[:, None] * m2 + q[None, :],
            output, mask=(p[:, None] < m1) & (q[None, :] < m2),
        )


    @triton.jit
    def _input_pair_backward_kernel(
        grad_ptr, U1_ptr, U2_ptr, grad_first_ptr, grad_x_ptr,
        n1: tl.constexpr, n2: tl.constexpr,
        r1: tl.constexpr, r2: tl.constexpr,
        BN1: tl.constexpr, BN2: tl.constexpr,
        BR1: tl.constexpr, BR2: tl.constexpr,
    ):
        token = tl.program_id(0)
        i, j = tl.arange(0, BN1), tl.arange(0, BN2)
        a, b = tl.arange(0, BR1), tl.arange(0, BR2)
        grad = tl.load(
            grad_ptr + token * r1 * r2 + a[:, None] * r2 + b[None, :],
            mask=(a[:, None] < r1) & (b[None, :] < r2), other=0.0,
        )
        U2_t = tl.load(
            U2_ptr + j[None, :] * r2 + b[:, None],
            mask=(b[:, None] < r2) & (j[None, :] < n2), other=0.0,
        )
        grad_first = tl.dot(grad, U2_t, input_precision="ieee")
        tl.store(
            grad_first_ptr + token * r1 * n2 + a[:, None] * n2 + j[None, :],
            grad_first, mask=(a[:, None] < r1) & (j[None, :] < n2),
        )
        U1 = tl.load(
            U1_ptr + i[:, None] * r1 + a[None, :],
            mask=(i[:, None] < n1) & (a[None, :] < r1), other=0.0,
        )
        grad_x = tl.dot(U1, grad_first.to(U1.dtype), input_precision="ieee")
        tl.store(
            grad_x_ptr + token * n1 * n2 + i[:, None] * n2 + j[None, :],
            grad_x, mask=(i[:, None] < n1) & (j[None, :] < n2),
        )


    @triton.jit
    def _output_pair_backward_kernel(
        grad_ptr, U3_ptr, U4_ptr, grad_third_ptr, grad_x_ptr,
        r3: tl.constexpr, r4: tl.constexpr,
        m1: tl.constexpr, m2: tl.constexpr,
        BR3: tl.constexpr, BR4: tl.constexpr,
        BM1: tl.constexpr, BM2: tl.constexpr,
    ):
        token = tl.program_id(0)
        c, d = tl.arange(0, BR3), tl.arange(0, BR4)
        p, q = tl.arange(0, BM1), tl.arange(0, BM2)
        grad = tl.load(
            grad_ptr + token * m1 * m2 + p[:, None] * m2 + q[None, :],
            mask=(p[:, None] < m1) & (q[None, :] < m2), other=0.0,
        )
        U4 = tl.load(
            U4_ptr + q[:, None] * r4 + d[None, :],
            mask=(q[:, None] < m2) & (d[None, :] < r4), other=0.0,
        )
        grad_third = tl.dot(grad, U4, input_precision="ieee")
        tl.store(
            grad_third_ptr + token * m1 * r4 + p[:, None] * r4 + d[None, :],
            grad_third, mask=(p[:, None] < m1) & (d[None, :] < r4),
        )
        U3_t = tl.load(
            U3_ptr + p[None, :] * r3 + c[:, None],
            mask=(c[:, None] < r3) & (p[None, :] < m1), other=0.0,
        )
        grad_x = tl.dot(U3_t, grad_third.to(U3_t.dtype), input_precision="ieee")
        tl.store(
            grad_x_ptr + token * r3 * r4 + c[:, None] * r4 + d[None, :],
            grad_x, mask=(c[:, None] < r3) & (d[None, :] < r4),
        )


def _ce_sum_and_grad(logits, targets, ignore_index, smoothing, *, gradients):
    """Use Liger normally and a host-sync-free fallback during graph capture."""
    if not torch.cuda.is_current_stream_capturing():
        return _fused_ce_sum_and_grad(
            logits,
            targets,
            ignore_index,
            smoothing,
            gradients=gradients,
        )

    # Liger 0.8.1 calls ``target_mask.sum().item()``, which is illegal while a
    # CUDA stream is being captured.  This path keeps every decision on device.
    logits_f = logits.float()
    valid = targets != ignore_index
    safe_targets = targets.masked_fill(~valid, 0)
    logsumexp = torch.logsumexp(logits_f, dim=-1)
    target_logits = logits_f.gather(1, safe_targets[:, None]).squeeze(1)
    nll = logsumexp - target_logits
    smooth_loss = logsumexp - logits_f.mean(dim=-1)
    loss = torch.where(
        valid,
        (1.0 - smoothing) * nll + smoothing * smooth_loss,
        0.0,
    ).sum()
    if not gradients:
        return loss, None
    grad = torch.softmax(logits_f, dim=-1)
    grad.sub_(smoothing / logits.shape[-1])
    grad.mul_(valid[:, None])
    _subtract_ce_target_kernel[(targets.numel(),)](
        grad,
        targets,
        columns=logits.shape[-1],
        ignore_index=ignore_index,
        scale=1.0 - smoothing,
        num_warps=1,
    )
    return loss, grad


def _cached_work_tensors(module, dtype: torch.dtype):
    parameters = (
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
    )
    key = (
        dtype,
        tuple((parameter.data_ptr(), parameter._version) for parameter in parameters),
    )
    cached = getattr(module, "_direct_tucker_work_cache", None)
    if cached is None or cached[0] != key:
        with torch.no_grad():
            work = tuple(parameter.to(dtype=dtype) for parameter in parameters)
        cached = (key, work)
        module._direct_tucker_work_cache = cached
    return cached[1]


def clear_work_caches(model) -> None:
    """Drop all experimental BF16 parameter caches in ``model``."""
    for module in model.modules():
        if hasattr(module, "_direct_tucker_work_cache"):
            del module._direct_tucker_work_cache
        if hasattr(module, "_paired_tucker_work_cache"):
            del module._paired_tucker_work_cache


def _block(value: int) -> int:
    return max(16, triton.next_power_of_2(value))


def _input_pair_saved(x, U1, U2):
    tokens, n1, n2 = x.shape
    r1, r2 = U1.shape[1], U2.shape[1]
    first = torch.empty((tokens, r1, n2), device=x.device, dtype=x.dtype)
    output = torch.empty((tokens, r1, r2), device=x.device, dtype=x.dtype)
    blocks = tuple(_block(v) for v in (n1, n2, r1, r2))
    _input_pair_saved_kernel[(tokens,)](
        x, U1, U2, first, output,
        n1=n1, n2=n2, r1=r1, r2=r2,
        BN1=blocks[0], BN2=blocks[1], BR1=blocks[2], BR2=blocks[3],
        num_warps=8 if max(n1, n2, r1, r2) > 32 else 4,
    )
    return output, first


def _output_pair_saved(x, U3, U4):
    tokens, r3, r4 = x.shape
    m1, m2 = U3.shape[0], U4.shape[0]
    third = torch.empty((tokens, m1, r4), device=x.device, dtype=x.dtype)
    output = torch.empty((tokens, m1, m2), device=x.device, dtype=x.dtype)
    blocks = tuple(_block(v) for v in (r3, r4, m1, m2))
    _output_pair_saved_kernel[(tokens,)](
        x, U3, U4, third, output,
        r3=r3, r4=r4, m1=m1, m2=m2,
        BR3=blocks[0], BR4=blocks[1], BM1=blocks[2], BM2=blocks[3],
        num_warps=8 if max(r3, r4, m1, m2) > 32 else 4,
    )
    return output, third


def _input_pair_backward(grad, U1, U2):
    tokens, r1, r2 = grad.shape
    n1, n2 = U1.shape[0], U2.shape[0]
    grad_first = torch.empty((tokens, r1, n2), device=grad.device, dtype=grad.dtype)
    grad_x = torch.empty((tokens, n1, n2), device=grad.device, dtype=grad.dtype)
    blocks = tuple(_block(v) for v in (n1, n2, r1, r2))
    _input_pair_backward_kernel[(tokens,)](
        grad, U1, U2, grad_first, grad_x,
        n1=n1, n2=n2, r1=r1, r2=r2,
        BN1=blocks[0], BN2=blocks[1], BR1=blocks[2], BR2=blocks[3],
        num_warps=8 if max(n1, n2, r1, r2) > 32 else 4,
    )
    return grad_x, grad_first


def _output_pair_backward(grad, U3, U4):
    tokens, m1, m2 = grad.shape
    r3, r4 = U3.shape[1], U4.shape[1]
    grad_third = torch.empty((tokens, m1, r4), device=grad.device, dtype=grad.dtype)
    grad_x = torch.empty((tokens, r3, r4), device=grad.device, dtype=grad.dtype)
    blocks = tuple(_block(v) for v in (r3, r4, m1, m2))
    _output_pair_backward_kernel[(tokens,)](
        grad, U3, U4, grad_third, grad_x,
        r3=r3, r4=r4, m1=m1, m2=m2,
        BR3=blocks[0], BR4=blocks[1], BM1=blocks[2], BM2=blocks[3],
        num_warps=8 if max(r3, r4, m1, m2) > 32 else 4,
    )
    return grad_x, grad_third


def _forward_chunk_saved_fast(x, core, U1, U2, U3, U4):
    """Recompute forward and retain the two factor-gradient intermediates."""
    tokens = x.shape[0]
    shaped = x.reshape(tokens, U1.shape[0], U2.shape[0])
    if not (can_fuse_input(U1, U2) and can_fuse_output(U3, U4)):
        return _forward_chunk(x, core, U1, U2, U3, U4)
    contracted, first = _input_pair_saved(shaped, U1, U2)
    core_out = F.linear(contracted.flatten(1), core).reshape(
        tokens, U3.shape[1], U4.shape[1]
    )
    output, third = _output_pair_saved(core_out, U3, U4)
    return output.flatten(1), (shaped, first, contracted, core_out, third)


def _backward_chunk_fast(grad_output, saved, core, U1, U2, U3, U4):
    if not (can_fuse_input(U1, U2) and can_fuse_output(U3, U4)):
        return _backward_chunk(grad_output, saved, core, U1, U2, U3, U4)
    shaped, first, contracted, core_out, third = saved
    tokens = shaped.shape[0]
    m1, m2 = U3.shape[0], U4.shape[0]
    r1, r2 = U1.shape[1], U2.shape[1]
    n1, n2 = U1.shape[0], U2.shape[0]
    r3, r4 = U3.shape[1], U4.shape[1]
    grad_output = grad_output.reshape(tokens, m1, m2)

    grad_core_out, grad_third = _output_pair_backward(grad_output, U3, U4)
    grad_U4 = grad_output.reshape(tokens * m1, m2).mT @ third.reshape(
        tokens * m1, r4
    )
    grouped_grad_third = grad_third.permute(0, 2, 1).reshape(tokens * r4, m1)
    grouped_core_out = core_out.permute(0, 2, 1).reshape(tokens * r4, r3)
    grad_U3 = grouped_grad_third.mT @ grouped_core_out

    grad_core_flat = grad_core_out.flatten(1)
    grad_core = grad_core_flat.mT @ contracted.flatten(1)
    grad_contracted = (grad_core_flat @ core).reshape(tokens, r1, r2)

    grad_input, grad_first = _input_pair_backward(grad_contracted, U1, U2)
    grad_U2 = first.reshape(tokens * r1, n2).mT @ grad_contracted.reshape(
        tokens * r1, r2
    )
    grouped_grad_first = grad_first.permute(0, 2, 1).reshape(tokens * n2, r1)
    grouped_input = shaped.permute(0, 2, 1).reshape(tokens * n2, n1)
    grad_U1 = grouped_input.mT @ grouped_grad_first
    return (
        grad_input.flatten(1), grad_core, grad_U1, grad_U2, grad_U3, grad_U4
    )


def _forward_chunk_fast(x, core, U1, U2, U3, U4):
    """Forward chunk with fused small mode pairs at any token count."""
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, r3 = U3.shape
    m2, r4 = U4.shape
    tokens = x.shape[0]
    shaped = x.reshape(tokens, n1, n2)

    if can_fuse_input(U1, U2):
        contracted_in = input_mode_pair(shaped, U1, U2)
    else:
        first = (
            shaped.permute(0, 2, 1).reshape(tokens * n2, n1) @ U1
        ).reshape(tokens, n2, r1).permute(0, 2, 1).contiguous()
        contracted_in = (first.reshape(tokens * r1, n2) @ U2).reshape(
            tokens, r1, r2
        )

    core_out = F.linear(contracted_in.flatten(1), core).reshape(
        tokens, r3, r4
    )
    if can_fuse_output(U3, U4):
        output = output_mode_pair(core_out, U3, U4)
    else:
        third = (
            core_out.permute(0, 2, 1).reshape(tokens * r4, r3) @ U3.mT
        ).reshape(tokens, r4, m1).permute(0, 2, 1).contiguous()
        output = (third.reshape(tokens * m1, r4) @ U4.mT).reshape(
            tokens, m1, m2
        )
    return output.flatten(1)


class FusedModeTuckerLinearFunction(torch.autograd.Function):
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
    ):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        outputs = []
        for start in range(0, flat_x.shape[0], int(chunk_size)):
            outputs.append(
                _forward_chunk_fast(
                    flat_x[start : start + int(chunk_size)],
                    corew,
                    U1w,
                    U2w,
                    U3w,
                    U4w,
                )
            )
        flat_output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)
        ctx.save_for_backward(x, corew, U1w, U2w, U3w, U4w)
        ctx.chunk_size = int(chunk_size)
        ctx.original_shape = original_shape
        ctx.output_features = U3.shape[0] * U4.shape[0]
        ctx.parameter_dtypes = tuple(
            parameter.dtype for parameter in (core, U1, U2, U3, U4)
        )
        return flat_output.reshape(*original_shape[:-1], ctx.output_features)

    @staticmethod
    def backward(ctx, grad_output):
        x, corew, U1w, U2w, U3w, U4w = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad_output = grad_output.reshape(-1, ctx.output_features)
        grad_x = torch.empty_like(flat_x)
        accumulators = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in (corew, U1w, U2w, U3w, U4w)
        ]
        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_x.shape[0])
            recompute = (
                _forward_chunk_saved_fast
                if _ENABLE_UNVALIDATED_FUSED_BACKWARD
                else _forward_chunk
            )
            backward_chunk = (
                _backward_chunk_fast
                if _ENABLE_UNVALIDATED_FUSED_BACKWARD
                else _backward_chunk
            )
            _, saved = recompute(flat_x[start:stop], corew, U1w, U2w, U3w, U4w)
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
        )


class FusedModeTuckerCrossEntropyFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        targets,
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
        ignore_index,
        label_smoothing,
    ):
        flat_x = x.reshape(-1, x.shape[-1])
        flat_targets = targets.reshape(-1)
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        valid_count = (flat_targets != int(ignore_index)).sum()
        for start in range(0, flat_x.shape[0], int(chunk_size)):
            stop = min(start + int(chunk_size), flat_x.shape[0])
            logits = _forward_chunk_fast(
                flat_x[start:stop], corew, U1w, U2w, U3w, U4w
            )
            fused_loss, _ = _ce_sum_and_grad(
                logits,
                flat_targets[start:stop],
                int(ignore_index),
                float(label_smoothing),
                gradients=False,
            )
            if fused_loss is None:
                fused_loss = F.cross_entropy(
                    logits.float(),
                    flat_targets[start:stop],
                    ignore_index=int(ignore_index),
                    label_smoothing=float(label_smoothing),
                    reduction="sum",
                )
            loss_sum.add_(fused_loss)
        ctx.save_for_backward(
            x, flat_targets, valid_count, corew, U1w, U2w, U3w, U4w
        )
        ctx.chunk_size = int(chunk_size)
        ctx.ignore_index = int(ignore_index)
        ctx.label_smoothing = float(label_smoothing)
        ctx.parameter_dtypes = tuple(
            parameter.dtype for parameter in (core, U1, U2, U3, U4)
        )
        return loss_sum / valid_count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_loss):
        (
            x,
            targets,
            valid_count,
            corew,
            U1w,
            U2w,
            U3w,
            U4w,
        ) = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        vocab_size = U3w.shape[0] * U4w.shape[0]
        grad_x = torch.empty_like(flat_x)
        accumulators = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in (corew, U1w, U2w, U3w, U4w)
        ]
        denominator = valid_count.clamp_min(1).float()
        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_x.shape[0])
            recompute = (
                _forward_chunk_saved_fast
                if _ENABLE_UNVALIDATED_FUSED_BACKWARD
                else _forward_chunk
            )
            backward_chunk = (
                _backward_chunk_fast
                if _ENABLE_UNVALIDATED_FUSED_BACKWARD
                else _backward_chunk
            )
            logits, saved = recompute(
                flat_x[start:stop], corew, U1w, U2w, U3w, U4w
            )
            chunk_targets = targets[start:stop]
            _, grad_logits = _ce_sum_and_grad(
                logits,
                chunk_targets,
                ctx.ignore_index,
                ctx.label_smoothing,
                gradients=True,
            )
            if grad_logits is None:
                valid = chunk_targets != ctx.ignore_index
                grad_logits = torch.softmax(logits.float(), dim=-1)
                grad_logits.sub_(ctx.label_smoothing / vocab_size)
                if valid.any():
                    rows = torch.arange(stop - start, device=x.device)[valid]
                    grad_logits[rows, chunk_targets[valid]] -= (
                        1.0 - ctx.label_smoothing
                    )
                grad_logits[~valid] = 0
            grad_logits.mul_(grad_loss.to(grad_logits.dtype) / denominator)
            grads = backward_chunk(
                grad_logits.to(dtype=flat_x.dtype),
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
            grad_x.reshape_as(x),
            None,
            *parameter_grads,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def fused_mode_tucker_linear(x, module, chunk_size: int):
    work = _cached_work_tensors(module, x.dtype)
    return FusedModeTuckerLinearFunction.apply(
        x,
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
        *work,
        chunk_size,
    )


def fused_mode_tucker_cross_entropy(
    x,
    targets,
    module,
    chunk_size: int,
    *,
    ignore_index: int = -1,
    label_smoothing: float = 0.0,
):
    work = _cached_work_tensors(module, x.dtype)
    if _ENABLE_ONLINE_CE:
        from tucker_online_ce import online_tucker_cross_entropy

        return online_tucker_cross_entropy(
            x,
            targets,
            module,
            work,
            chunk_size,
            _ONLINE_CE_OUTPUT_TILE,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing,
        )
    return FusedModeTuckerCrossEntropyFunction.apply(
        x,
        targets,
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
        *work,
        chunk_size,
        ignore_index,
        label_smoothing,
    )


def install(
    *,
    fused_backward: bool | None = None,
    online_ce: bool | None = None,
    output_mode_tile: int | None = None,
) -> None:
    """Install the experimental functions into the dynamic model call sites."""
    global _ENABLE_UNVALIDATED_FUSED_BACKWARD
    global _ENABLE_ONLINE_CE
    global _ONLINE_CE_OUTPUT_TILE

    if fused_backward is not None:
        _ENABLE_UNVALIDATED_FUSED_BACKWARD = bool(fused_backward)
    if online_ce is not None:
        _ENABLE_ONLINE_CE = bool(online_ce)
    if output_mode_tile is not None:
        if int(output_mode_tile) <= 0:
            raise ValueError("online CE output-mode tile must be positive")
        _ONLINE_CE_OUTPUT_TILE = int(output_mode_tile)
    import models.tucker_chunked as baseline

    baseline.chunked_tucker_linear = fused_mode_tucker_linear
    baseline.chunked_tucker_cross_entropy = fused_mode_tucker_cross_entropy
