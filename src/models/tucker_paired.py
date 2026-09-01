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
    r1, r2 = U1.shape[1], U2.shape[1]
    r3 = U3.shape[1]
    core = core.reshape(r3, r1, r2).permute(1, 2, 0)
    paired = torch.einsum("abc,xa,yb,zc->xyz", core, U1, U2, U3)
    return paired_tensor_to_weight(paired, input_modes, output_modes)


class PairedTuckerLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, core, U1, U2, U3, input_modes, output_modes):
        work = tuple(parameter.to(dtype=x.dtype) for parameter in (core, U1, U2, U3))
        weight = _materialize_paired_weight(
            *work,
            input_modes,
            output_modes,
        )
        ctx.save_for_backward(x, core, U1, U2, U3)
        ctx.input_modes = input_modes
        ctx.output_modes = output_modes
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, core, U1, U2, U3 = ctx.saved_tensors
        corew, U1w, U2w, U3w = tuple(
            parameter.to(dtype=x.dtype) for parameter in (core, U1, U2, U3)
        )
        weight = _materialize_paired_weight(
            corew,
            U1w,
            U2w,
            U3w,
            ctx.input_modes,
            ctx.output_modes,
        )
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

        r1, r2 = U1w.shape[1], U2w.shape[1]
        r3 = U3w.shape[1]
        core_tensor = corew.reshape(r3, r1, r2).permute(1, 2, 0)
        projected_c = torch.einsum("xyz,zc->xyc", grad_paired, U3w)
        projected_bc = torch.einsum("xyc,yb->xbc", projected_c, U2w)
        grad_core = torch.einsum("xbc,xa->abc", projected_bc, U1w)
        grad_U1 = torch.einsum("xbc,abc->xa", projected_bc, core_tensor)

        projected_ac = torch.einsum("xyc,xa->ayc", projected_c, U1w)
        grad_U2 = torch.einsum("ayc,abc->yb", projected_ac, core_tensor)

        projected_bz = torch.einsum("xyz,yb->xbz", grad_paired, U2w)
        projected_abz = torch.einsum("xbz,xa->abz", projected_bz, U1w)
        grad_U3 = torch.einsum("abz,abc->zc", projected_abz, core_tensor)
        grad_core = grad_core.permute(2, 0, 1).reshape_as(core)
        return (
            grad_x.reshape_as(x),
            grad_core.to(dtype=core.dtype),
            grad_U1.to(dtype=U1.dtype),
            grad_U2.to(dtype=U2.dtype),
            grad_U3.to(dtype=U3.dtype),
            None,
            None,
        )


def paired_tucker_linear(x, module):
    work_dtype = (
        torch.get_autocast_dtype(x.device.type)
        if torch.is_autocast_enabled(x.device.type)
        else x.dtype
    )
    return PairedTuckerLinearFunction.apply(
        x.to(dtype=work_dtype),
        module.core_matrix,
        module.U1,
        module.U2,
        module.U3,
        module.paired_in_modes,
        module.paired_out_modes,
    )
