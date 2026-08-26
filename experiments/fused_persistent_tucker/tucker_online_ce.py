"""Online cross entropy for a direct Tucker language-model head.

The output factors are absorbed into a *vocabulary tile* of the Tucker core.
Only that projected-core tile and its logits are materialised; neither the full
dense weight nor a ``[token_chunk, vocab_size]`` logits tensor is constructed.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - CPU-only development hosts
    triton = None
    tl = None

from models.tucker_triton import can_fuse_input, input_mode_pair


if triton is not None:

    @triton.jit
    def _subtract_target_tile_kernel(
        grad_ptr,
        target_ptr,
        rows: tl.constexpr,
        columns: tl.constexpr,
        vocabulary_start: tl.constexpr,
        ignore_index: tl.constexpr,
        scale_ptr,
    ):
        row = tl.program_id(0)
        target = tl.load(target_ptr + row)
        local_target = target - vocabulary_start
        valid = (
            (row < rows)
            & (target != ignore_index)
            & (local_target >= 0)
            & (local_target < columns)
        )
        offset = row * columns + local_target
        value = tl.load(grad_ptr + offset, mask=valid, other=0.0)
        scale = tl.load(scale_ptr)
        tl.store(grad_ptr + offset, value - scale, mask=valid)


def _input_forward(x, U1, U2, *, save_first: bool):
    """Apply the two input modes and optionally retain the U1 intermediate."""
    tokens = x.shape[0]
    n1, r1 = U1.shape
    n2, r2 = U2.shape
    shaped = x.reshape(tokens, n1, n2)
    if can_fuse_input(U1, U2) and not save_first:
        return input_mode_pair(shaped, U1, U2), shaped, None
    first = (
        shaped.permute(0, 2, 1).reshape(tokens * n2, n1) @ U1
    ).reshape(tokens, n2, r1).permute(0, 2, 1).contiguous()
    contracted = (first.reshape(tokens * r1, n2) @ U2).reshape(
        tokens, r1, r2
    )
    return contracted, shaped, first


def _input_backward(grad_contracted, shaped, first, U1, U2):
    """Analytical VJP for the two input modes."""
    tokens = shaped.shape[0]
    n1, r1 = U1.shape
    n2, r2 = U2.shape
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
    return grad_input.flatten(1), grad_U1, grad_U2


def _projected_core_tile(core, U3_tile, U4):
    """Return one ``[vocabulary_tile, input_rank]`` projected-core tile.

    This is a partial contraction of the Tucker core, not the full logical
    ``[vocab_size, hidden_size]`` dense weight.  Input factors U1/U2 are not
    absorbed into it.
    """
    r3 = U3_tile.shape[1]
    r4 = U4.shape[1]
    input_rank = core.shape[1]
    core_flat = core.reshape(r3, r4 * input_rank)
    after_U3 = U3_tile @ core_flat
    after_U3_3d = after_U3.reshape(U3_tile.shape[0], r4, input_rank)
    projected = torch.matmul(U4.unsqueeze(0), after_U3_3d)
    return projected.reshape(-1, input_rank), after_U3_3d, core_flat


def _online_statistics(
    contracted,
    targets,
    core,
    U3,
    U4,
    *,
    token_chunk_size,
    output_mode_tile,
    ignore_index,
):
    """Compute per-token logsumexp, target logit, and logits sum online."""
    tokens = contracted.shape[0]
    contracted = contracted.flatten(1)
    m1, m2 = U3.shape[0], U4.shape[0]
    row_max = torch.full(
        (tokens,), -float("inf"), device=contracted.device, dtype=torch.float32
    )
    row_sum = torch.zeros_like(row_max)
    target_logits = torch.zeros_like(row_max)
    logits_sum = torch.zeros_like(row_max)

    for p_start in range(0, m1, output_mode_tile):
        p_stop = min(p_start + output_mode_tile, m1)
        vocabulary_start = p_start * m2
        vocabulary_stop = p_stop * m2
        weight_tile, _, _ = _projected_core_tile(
            core, U3[p_start:p_stop], U4
        )
        for token_start in range(0, tokens, token_chunk_size):
            token_stop = min(token_start + token_chunk_size, tokens)
            logits = contracted[token_start:token_stop] @ weight_tile.mT
            logits_f = logits.float()
            old_max = row_max[token_start:token_stop]
            old_sum = row_sum[token_start:token_stop]
            tile_max = logits_f.amax(dim=1)
            new_max = torch.maximum(old_max, tile_max)
            new_sum = old_sum * torch.exp(old_max - new_max)
            new_sum.add_(torch.exp(logits_f - new_max[:, None]).sum(dim=1))
            row_max[token_start:token_stop] = new_max
            row_sum[token_start:token_stop] = new_sum
            logits_sum[token_start:token_stop].add_(logits_f.sum(dim=1))

            chunk_targets = targets[token_start:token_stop]
            in_tile = (
                (chunk_targets >= vocabulary_start)
                & (chunk_targets < vocabulary_stop)
                & (chunk_targets != ignore_index)
            )
            local_target = (chunk_targets - vocabulary_start).clamp(
                0, vocabulary_stop - vocabulary_start - 1
            )
            selected = logits_f.gather(1, local_target[:, None]).squeeze(1)
            current = target_logits[token_start:token_stop]
            target_logits[token_start:token_stop] = torch.where(
                in_tile, selected, current
            )
    return row_max + torch.log(row_sum), target_logits, logits_sum


class OnlineTuckerCrossEntropyFunction(torch.autograd.Function):
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
        token_chunk_size,
        output_mode_tile,
        ignore_index,
        label_smoothing,
    ):
        original_shape = x.shape
        flat_x = x.reshape(-1, original_shape[-1])
        flat_targets = targets.reshape(-1)
        contracted, _, _ = _input_forward(flat_x, U1w, U2w, save_first=False)
        lse, target_logits, logits_sum = _online_statistics(
            contracted,
            flat_targets,
            corew,
            U3w,
            U4w,
            token_chunk_size=int(token_chunk_size),
            output_mode_tile=int(output_mode_tile),
            ignore_index=int(ignore_index),
        )
        valid = flat_targets != int(ignore_index)
        valid_count = valid.sum()
        vocab_size = U3w.shape[0] * U4w.shape[0]
        nll = lse - target_logits
        smooth_loss = lse - logits_sum / vocab_size
        per_token = (1.0 - float(label_smoothing)) * nll
        per_token.add_(float(label_smoothing) * smooth_loss)
        loss = torch.where(valid, per_token, 0.0).sum()

        ctx.save_for_backward(
            x, flat_targets, lse, valid_count, corew, U1w, U2w, U3w, U4w
        )
        ctx.token_chunk_size = int(token_chunk_size)
        ctx.output_mode_tile = int(output_mode_tile)
        ctx.ignore_index = int(ignore_index)
        ctx.label_smoothing = float(label_smoothing)
        ctx.original_shape = original_shape
        ctx.parameter_dtypes = tuple(
            parameter.dtype for parameter in (core, U1, U2, U3, U4)
        )
        return loss / valid_count.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_loss):
        (
            x,
            targets,
            lse,
            valid_count,
            core,
            U1,
            U2,
            U3,
            U4,
        ) = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        contracted, shaped, first = _input_forward(
            flat_x, U1, U2, save_first=True
        )
        tokens = contracted.shape[0]
        input_rank = contracted.shape[1] * contracted.shape[2]
        contracted_flat = contracted.reshape(tokens, input_rank)
        m1, m2 = U3.shape[0], U4.shape[0]
        vocab_size = m1 * m2

        grad_contracted = torch.zeros_like(contracted_flat, dtype=torch.float32)
        grad_core = torch.zeros_like(core, dtype=torch.float32)
        grad_U3 = torch.zeros_like(U3, dtype=torch.float32)
        grad_U4 = torch.zeros_like(U4, dtype=torch.float32)
        denominator = valid_count.clamp_min(1).float()
        scalar_scale = grad_loss.float() / denominator

        for p_start in range(0, m1, ctx.output_mode_tile):
            p_stop = min(p_start + ctx.output_mode_tile, m1)
            rows_in_tile = p_stop - p_start
            vocabulary_start = p_start * m2
            vocabulary_columns = rows_in_tile * m2
            U3_tile = U3[p_start:p_stop]
            weight_tile, after_U3, core_flat = _projected_core_tile(
                core, U3_tile, U4
            )
            grad_weight = torch.zeros_like(weight_tile, dtype=torch.float32)

            for token_start in range(0, tokens, ctx.token_chunk_size):
                token_stop = min(token_start + ctx.token_chunk_size, tokens)
                contracted_chunk = contracted_flat[token_start:token_stop]
                logits = contracted_chunk @ weight_tile.mT
                grad_logits = torch.exp(
                    logits.float() - lse[token_start:token_stop, None]
                )
                grad_logits.sub_(ctx.label_smoothing / vocab_size)
                chunk_targets = targets[token_start:token_stop]
                valid = chunk_targets != ctx.ignore_index
                grad_logits.mul_(valid[:, None])
                target_scale = torch.as_tensor(
                    (1.0 - ctx.label_smoothing),
                    device=grad_logits.device,
                    dtype=grad_logits.dtype,
                )
                if triton is not None and grad_logits.is_cuda:
                    _subtract_target_tile_kernel[(token_stop - token_start,)](
                        grad_logits,
                        chunk_targets,
                        rows=token_stop - token_start,
                        columns=vocabulary_columns,
                        vocabulary_start=vocabulary_start,
                        ignore_index=ctx.ignore_index,
                        scale_ptr=target_scale,
                        num_warps=1,
                    )
                else:  # pragma: no cover - CUDA is required for benchmarks
                    in_tile = (
                        (chunk_targets >= vocabulary_start)
                        & (chunk_targets < vocabulary_start + vocabulary_columns)
                        & valid
                    )
                    local = (chunk_targets - vocabulary_start).clamp(
                        0, vocabulary_columns - 1
                    )
                    grad_logits.scatter_add_(
                        1,
                        local[:, None],
                        (-(1.0 - ctx.label_smoothing) * in_tile.float())[:, None],
                    )
                grad_logits.mul_(scalar_scale)
                grad_logits_work = grad_logits.to(dtype=contracted.dtype)
                grad_contracted[token_start:token_stop].add_(
                    (grad_logits_work @ weight_tile).float()
                )
                grad_weight.add_(
                    (grad_logits_work.mT @ contracted_chunk).float()
                )

            grad_weight_work = grad_weight.to(dtype=contracted.dtype).reshape(
                rows_in_tile, m2, input_rank
            )
            grad_U4.add_(
                torch.bmm(grad_weight_work, after_U3.mT).sum(dim=0).float()
            )
            grad_after_U3 = torch.matmul(U4.mT.unsqueeze(0), grad_weight_work)
            grad_after_U3_flat = grad_after_U3.reshape(rows_in_tile, -1)
            grad_U3[p_start:p_stop].add_(
                (grad_after_U3_flat @ core_flat.mT).float()
            )
            grad_core.add_(
                (U3_tile.mT @ grad_after_U3_flat).reshape_as(core).float()
            )

        grad_input, grad_U1, grad_U2 = _input_backward(
            grad_contracted.to(dtype=contracted.dtype).reshape_as(contracted),
            shaped,
            first,
            U1,
            U2,
        )
        parameter_grads = tuple(
            gradient.to(dtype=dtype)
            for gradient, dtype in zip(
                (grad_core, grad_U1, grad_U2, grad_U3, grad_U4),
                ctx.parameter_dtypes,
            )
        )
        return (
            grad_input.reshape(ctx.original_shape),
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
            None,
        )


def online_tucker_cross_entropy(
    x,
    targets,
    module,
    work_tensors,
    token_chunk_size,
    output_mode_tile,
    *,
    ignore_index=-1,
    label_smoothing=0.0,
):
    """Apply online CE with explicitly supplied BF16 work tensors."""
    return OnlineTuckerCrossEntropyFunction.apply(
        x,
        targets,
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.U4,
        *work_tensors,
        int(token_chunk_size),
        int(output_mode_tile),
        int(ignore_index),
        float(label_smoothing),
    )
