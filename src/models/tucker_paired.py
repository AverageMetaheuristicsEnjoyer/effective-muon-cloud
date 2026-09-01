from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def paired_tensor_to_weight(
    paired: torch.Tensor,
    input_modes: tuple[int, int, int],
    output_modes: tuple[int, int, int],
) -> torch.Tensor:
    o1, o2, o3 = output_modes
    i1, i2, i3 = input_modes
    return (
        paired.reshape(o1, i1, o2, i2, o3, i3)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(math.prod(output_modes), math.prod(input_modes))
    )


def weight_to_paired_tensor(
    weight: torch.Tensor,
    input_modes: tuple[int, int, int],
    output_modes: tuple[int, int, int],
) -> torch.Tensor:
    o1, o2, o3 = output_modes
    i1, i2, i3 = input_modes
    return (
        weight.reshape(o1, o2, o3, i1, i2, i3)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(o1 * i1, o2 * i2, o3 * i3)
    )


def _materialize_paired_weight(
    core: torch.Tensor,
    U1: torch.Tensor,
    U2: torch.Tensor,
    U3: torch.Tensor,
    input_modes: tuple[int, int, int],
    output_modes: tuple[int, int, int],
) -> torch.Tensor:
    return _expand_paired_weight(
        core,
        U1,
        U2,
        U3,
        input_modes,
        output_modes,
    )[-1]


def _expand_paired_weight(
    core: torch.Tensor,
    U1: torch.Tensor,
    U2: torch.Tensor,
    U3: torch.Tensor,
    input_modes: tuple[int, int, int],
    output_modes: tuple[int, int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r1, r2 = U1.shape[1], U2.shape[1]
    r3 = U3.shape[1]
    p1, p2 = U1.shape[0], U2.shape[0]
    core_tensor = core.reshape(r3, r1, r2).permute(1, 2, 0)
    after_u1 = (U1 @ core_tensor.reshape(r1, r2 * r3)).reshape(p1, r2, r3)
    after_u2 = (
        after_u1.permute(0, 2, 1).reshape(p1 * r3, r2) @ U2.mT
    ).reshape(p1, r3, p2).permute(0, 2, 1).contiguous()
    paired = (after_u2.reshape(p1 * p2, r3) @ U3.mT).reshape(
        p1, p2, U3.shape[0]
    )
    return (
        after_u1,
        after_u2,
        paired_tensor_to_weight(paired, input_modes, output_modes),
    )


def _fresh_paired_work(module, dtype: torch.dtype):
    with torch.no_grad():
        core, U1, U2, U3 = tuple(
            parameter.to(dtype=dtype)
            for parameter in (module.core_matrix, module.U1, module.U2, module.U3)
        )
        after_u1, after_u2, weight = _expand_paired_weight(
            core,
            U1,
            U2,
            U3,
            module.paired_in_modes,
            module.paired_out_modes,
        )
    return core, U1, U2, U3, after_u1, after_u2, weight


def _cached_paired_work(module, dtype: torch.dtype):
    parameters = (module.core_matrix, module.U1, module.U2, module.U3)
    key = (
        dtype,
        tuple((parameter.data_ptr(), parameter._version) for parameter in parameters),
    )
    cached = getattr(module, "_paired_tucker_work_cache", None)
    if cached is None or cached[0] != key:
        cached = (key, _fresh_paired_work(module, dtype))
        module._paired_tucker_work_cache = cached
    return cached[1]


class PairedTuckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x,
        core,
        U1,
        U2,
        U3,
        corew,
        U1w,
        U2w,
        U3w,
        after_u1,
        after_u2,
        weight,
        input_modes,
        output_modes,
    ):
        ctx.save_for_backward(x, corew, U1w, U2w, U3w, after_u1, after_u2, weight)
        ctx.input_modes = input_modes
        ctx.output_modes = output_modes
        ctx.parameter_dtypes = tuple(
            parameter.dtype for parameter in (core, U1, U2, U3)
        )
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, corew, U1w, U2w, U3w, after_u1, after_u2, weight = ctx.saved_tensors
        flat_x = x.reshape(-1, x.shape[-1])
        flat_grad_output = grad_output.reshape(-1, grad_output.shape[-1]).to(
            dtype=x.dtype
        )
        grad_x = flat_grad_output @ weight
        grad_weight = flat_grad_output.mT @ flat_x
        grad_paired = weight_to_paired_tensor(
            grad_weight,
            ctx.input_modes,
            ctx.output_modes,
        )

        p1, p2 = U1w.shape[0], U2w.shape[0]
        r1, r2 = U1w.shape[1], U2w.shape[1]
        r3 = U3w.shape[1]
        core_tensor = corew.reshape(r3, r1, r2).permute(1, 2, 0)
        grad_paired_matrix = grad_paired.reshape(p1 * p2, U3w.shape[0])
        grad_U3 = grad_paired_matrix.mT @ after_u2.reshape(p1 * p2, r3)
        grad_after_u2 = (grad_paired_matrix @ U3w).reshape(p1, p2, r3)

        grouped_grad_after_u2 = grad_after_u2.permute(0, 2, 1).reshape(
            p1 * r3, p2
        )
        grouped_after_u1 = after_u1.permute(0, 2, 1).reshape(p1 * r3, r2)
        grad_U2 = grouped_grad_after_u2.mT @ grouped_after_u1
        grad_after_u1 = (grouped_grad_after_u2 @ U2w).reshape(
            p1, r3, r2
        ).permute(0, 2, 1).contiguous()

        flat_grad_after_u1 = grad_after_u1.reshape(p1, r2 * r3)
        flat_core = core_tensor.reshape(r1, r2 * r3)
        grad_U1 = flat_grad_after_u1 @ flat_core.mT
        grad_core = (U1w.mT @ flat_grad_after_u1).reshape(r1, r2, r3)
        grad_core = grad_core.permute(2, 0, 1).reshape_as(corew)
        parameter_grads = tuple(
            gradient.to(dtype=dtype)
            for gradient, dtype in zip(
                (grad_core, grad_U1, grad_U2, grad_U3),
                ctx.parameter_dtypes,
            )
        )
        return (
            grad_x.reshape_as(x),
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


def paired_tucker_linear(x, module, *, cache_policy: str = "recast"):
    work_dtype = (
        torch.get_autocast_dtype(x.device.type)
        if torch.is_autocast_enabled(x.device.type)
        else x.dtype
    )
    persistent = cache_policy == "persistent" or (
        cache_policy == "hybrid_gate_up" and module.out_features == 2816
    )
    work = (
        _cached_paired_work(module, work_dtype)
        if persistent
        else _fresh_paired_work(module, work_dtype)
    )
    return PairedTuckerLinearFunction.apply(
        x.to(dtype=work_dtype),
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        *work,
        module.paired_in_modes,
        module.paired_out_modes,
    )
