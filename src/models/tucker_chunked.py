"""Memory-bounded Tucker contractions with an analytical custom backward.

The implementation deliberately saves only the original input and Tucker
parameters.  Forward intermediates are produced one token chunk at a time and
discarded; backward recomputes the same chunk and accumulates exact gradients.
This is the correctness reference for the Triton kernels.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.tucker_triton import (
    can_fuse_input,
    can_fuse_output,
    input_mode_pair,
    output_mode_pair,
)

try:
    from liger_kernel.ops.cross_entropy import cross_entropy_forward as _liger_ce_forward
except ImportError:  # pragma: no cover - optional on CPU development hosts
    _liger_ce_forward = None


def _fused_ce_sum_and_grad(
    logits, targets, ignore_index, label_smoothing, *, gradients
):
    """Liger CE sum; optionally overwrite ``logits`` with d(loss_sum)/dlogits."""
    if _liger_ce_forward is None or not logits.is_cuda:
        return None, None
    if gradients:
        logits.requires_grad_(True)
    loss, _, _, _, grad_logits = _liger_ce_forward(
        logits,
        targets,
        None,
        ignore_index,
        0.0,
        label_smoothing,
        "sum",
        None,
        False,
    )
    return loss.float(), grad_logits if gradients else None


def _work_tensors(x, core, U1, U2, U3, U4):
    dtype = x.dtype
    return (
        x,
        core.to(dtype=dtype),
        U1.to(dtype=dtype),
        U2.to(dtype=dtype),
        U3.to(dtype=dtype),
        U4.to(dtype=dtype),
    )


def _forward_chunk(x, core, U1, U2, U3, U4, use_triton=False):
    """Contract a flat ``[tokens, in_features]`` chunk."""
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    m1, r3 = U3.shape
    m2, r4 = U4.shape
    shaped = x.reshape(-1, n1, n2)
    first = None
    tokens = shaped.shape[0]
    if use_triton and tokens <= 1024 and can_fuse_input(U1, U2):
        contracted_in = input_mode_pair(shaped, U1, U2)
    else:
        # Express the two mode products as large GEMMs. This is substantially
        # faster than generic einsum for thousands of tokens on A100.
        first = (
            shaped.permute(0, 2, 1).reshape(tokens * n2, n1) @ U1
        ).reshape(tokens, n2, r1).permute(0, 2, 1).contiguous()
        contracted_in = (first.reshape(tokens * r1, n2) @ U2).reshape(
            tokens, r1, r2
        )
    core_out = F.linear(contracted_in.flatten(1), core).reshape(-1, r3, r4)
    third = None
    if use_triton and tokens <= 1024 and can_fuse_output(U3, U4):
        output = output_mode_pair(core_out, U3, U4)
    else:
        third = (
            core_out.permute(0, 2, 1).reshape(tokens * r4, r3) @ U3.mT
        ).reshape(tokens, r4, m1).permute(0, 2, 1).contiguous()
        output = (third.reshape(tokens * m1, r4) @ U4.mT).reshape(
            tokens, m1, m2
        )
    # Backward recomputes the unfused reference intermediates; forward callers
    # do not retain this tuple.
    return output.flatten(1), (shaped, first, contracted_in, core_out, third)


def _backward_chunk(grad_output, saved, core, U1, U2, U3, U4):
    """Analytical VJP for one chunk; no autograd graph is constructed."""
    shaped, first, contracted_in, core_out, third = saved
    m1 = U3.shape[0]
    m2 = U4.shape[0]
    grad_output = grad_output.reshape(-1, m1, m2)
    tokens = grad_output.shape[0]
    r1, r2 = U1.shape[1], U2.shape[1]
    r3, r4 = U3.shape[1], U4.shape[1]
    n1, n2 = U1.shape[0], U2.shape[0]

    grad_output_flat = grad_output.reshape(tokens * m1, m2)
    third_flat = third.reshape(tokens * m1, r4)
    grad_third = (grad_output_flat @ U4).reshape(tokens, m1, r4)
    grad_U4 = grad_output_flat.mT @ third_flat
    grouped_grad_third = grad_third.permute(0, 2, 1).reshape(
        tokens * r4, m1
    )
    grouped_core_out = core_out.permute(0, 2, 1).reshape(tokens * r4, r3)
    grad_core_out = (grouped_grad_third @ U3).reshape(
        tokens, r4, r3
    ).permute(0, 2, 1).contiguous()
    grad_U3 = grouped_grad_third.mT @ grouped_core_out

    grad_core_flat = grad_core_out.flatten(1)
    contracted_flat = contracted_in.flatten(1)
    grad_core = grad_core_flat.mT @ contracted_flat
    grad_contracted = (grad_core_flat @ core).reshape_as(contracted_in)

    grad_contracted_flat = grad_contracted.reshape(tokens * r1, r2)
    first_flat = first.reshape(tokens * r1, n2)
    grad_first = (grad_contracted_flat @ U2.mT).reshape(tokens, r1, n2)
    grad_U2 = first_flat.mT @ grad_contracted_flat
    grouped_grad_first = grad_first.permute(0, 2, 1).reshape(tokens * n2, r1)
    grouped_input = shaped.permute(0, 2, 1).reshape(tokens * n2, n1)
    grad_input = (grouped_grad_first @ U1.mT).reshape(
        tokens, n2, n1
    ).permute(0, 2, 1).contiguous()
    grad_U1 = grouped_input.mT @ grouped_grad_first
    return (
        grad_input.flatten(1),
        grad_core,
        grad_U1,
        grad_U2,
        grad_U3,
        grad_U4,
    )


class ChunkedTuckerLinearFunction(torch.autograd.Function):
    """Tucker linear map that rematerializes chunk intermediates in backward."""

    @staticmethod
    def forward(ctx, x, core, U1, U2, U3, U4, chunk_size):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        chunk_size = int(chunk_size)
        outputs = []
        xw, corew, U1w, U2w, U3w, U4w = _work_tensors(
            flat_x, core, U1, U2, U3, U4
        )
        for start in range(0, flat_x.shape[0], chunk_size):
            output, _ = _forward_chunk(
                xw[start : start + chunk_size],
                corew,
                U1w,
                U2w,
                U3w,
                U4w,
                use_triton=True,
            )
            outputs.append(output)
        flat_output = torch.cat(outputs, dim=0)
        ctx.save_for_backward(x, core, U1, U2, U3, U4)
        ctx.chunk_size = chunk_size
        ctx.original_shape = original_shape
        ctx.output_features = U3.shape[0] * U4.shape[0]
        return flat_output.reshape(*original_shape[:-1], ctx.output_features)

    @staticmethod
    def backward(ctx, grad_output):
        x, core, U1, U2, U3, U4 = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad_output = grad_output.reshape(-1, ctx.output_features)
        xw, corew, U1w, U2w, U3w, U4w = _work_tensors(
            flat_x, core, U1, U2, U3, U4
        )

        grad_x = torch.empty_like(flat_x)
        grad_core = torch.zeros_like(core, dtype=torch.float32)
        grad_U1 = torch.zeros_like(U1, dtype=torch.float32)
        grad_U2 = torch.zeros_like(U2, dtype=torch.float32)
        grad_U3 = torch.zeros_like(U3, dtype=torch.float32)
        grad_U4 = torch.zeros_like(U4, dtype=torch.float32)
        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_x.shape[0])
            _, saved = _forward_chunk(
                xw[start:stop], corew, U1w, U2w, U3w, U4w
            )
            grads = _backward_chunk(
                flat_grad_output[start:stop].to(dtype=xw.dtype),
                saved,
                corew,
                U1w,
                U2w,
                U3w,
                U4w,
            )
            grad_x[start:stop].copy_(grads[0].to(dtype=grad_x.dtype))
            for accumulator, value in zip(
                (grad_core, grad_U1, grad_U2, grad_U3, grad_U4), grads[1:]
            ):
                accumulator.add_(value.float())
        return (
            grad_x.reshape(ctx.original_shape),
            grad_core.to(dtype=core.dtype),
            grad_U1.to(dtype=U1.dtype),
            grad_U2.to(dtype=U2.dtype),
            grad_U3.to(dtype=U3.dtype),
            grad_U4.to(dtype=U4.dtype),
            None,
        )


class ChunkedTuckerCrossEntropyFunction(torch.autograd.Function):
    """Fused Tucker head and CE without a full weight or logits tensor."""

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
        chunk_size,
        ignore_index,
        label_smoothing,
    ):
        flat_x = x.reshape(-1, x.shape[-1])
        targets = targets.reshape(-1)
        chunk_size = int(chunk_size)
        ignore_index = int(ignore_index)
        label_smoothing = float(label_smoothing)
        xw, corew, U1w, U2w, U3w, U4w = _work_tensors(
            flat_x, core, U1, U2, U3, U4
        )
        loss_sum = torch.zeros((), device=x.device, dtype=torch.float32)
        valid_count = (targets != ignore_index).sum()
        for start in range(0, flat_x.shape[0], chunk_size):
            stop = min(start + chunk_size, flat_x.shape[0])
            logits, _ = _forward_chunk(
                xw[start:stop],
                corew,
                U1w,
                U2w,
                U3w,
                U4w,
                use_triton=True,
            )
            chunk_targets = targets[start:stop]
            fused_loss, _ = _fused_ce_sum_and_grad(
                logits,
                chunk_targets,
                ignore_index,
                label_smoothing,
                gradients=False,
            )
            if fused_loss is not None:
                loss_sum += fused_loss
                continue
            logits = logits.float()
            valid = chunk_targets != ignore_index
            if valid.any():
                valid_logits = logits[valid]
                valid_targets = chunk_targets[valid]
                logsumexp = torch.logsumexp(valid_logits, dim=-1)
                target_logits = valid_logits.gather(
                    1, valid_targets[:, None]
                ).squeeze(1)
                nll = logsumexp - target_logits
                smooth = logsumexp - valid_logits.mean(dim=-1)
                loss_sum += (
                    (1.0 - label_smoothing) * nll
                    + label_smoothing * smooth
                ).sum()
        loss = loss_sum / valid_count.clamp_min(1)
        ctx.save_for_backward(x, targets, core, U1, U2, U3, U4, valid_count)
        ctx.chunk_size = chunk_size
        ctx.ignore_index = ignore_index
        ctx.label_smoothing = label_smoothing
        return loss

    @staticmethod
    def backward(ctx, grad_loss):
        x, targets, core, U1, U2, U3, U4, valid_count = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        xw, corew, U1w, U2w, U3w, U4w = _work_tensors(
            flat_x, core, U1, U2, U3, U4
        )
        vocab_size = U3.shape[0] * U4.shape[0]
        grad_x = torch.empty_like(flat_x)
        accumulators = [
            torch.zeros_like(parameter, dtype=torch.float32)
            for parameter in (core, U1, U2, U3, U4)
        ]
        denominator = valid_count.clamp_min(1).float()
        for start in range(0, flat_x.shape[0], ctx.chunk_size):
            stop = min(start + ctx.chunk_size, flat_x.shape[0])
            logits, saved = _forward_chunk(
                xw[start:stop], corew, U1w, U2w, U3w, U4w
            )
            chunk_targets = targets[start:stop]
            _, grad_logits = _fused_ce_sum_and_grad(
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
                    grad_logits[
                        rows, chunk_targets[valid]
                    ] -= 1.0 - ctx.label_smoothing
                grad_logits[~valid] = 0
            grad_logits.mul_(grad_loss.to(dtype=grad_logits.dtype) / denominator)
            grads = _backward_chunk(
                grad_logits.to(dtype=xw.dtype),
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
        return (
            grad_x.reshape_as(x),
            None,
            *(gradient.to(dtype=parameter.dtype) for gradient, parameter in zip(
                accumulators, (core, U1, U2, U3, U4)
            )),
            None,
            None,
            None,
        )


def chunked_tucker_linear(x, module, chunk_size: int):
    return ChunkedTuckerLinearFunction.apply(
        x,
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
        chunk_size,
    )


def chunked_tucker_cross_entropy(
    x,
    targets,
    module,
    chunk_size: int,
    *,
    ignore_index: int = -1,
    label_smoothing: float = 0.0,
):
    return ChunkedTuckerCrossEntropyFunction.apply(
        x,
        targets,
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
        chunk_size,
        ignore_index,
        label_smoothing,
    )
