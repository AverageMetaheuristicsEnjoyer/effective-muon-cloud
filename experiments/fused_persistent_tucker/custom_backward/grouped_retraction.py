"""Batched QR gauge fixing for Tucker modules."""

from __future__ import annotations

from collections import defaultdict

import torch

from models.tucker_linear import (
    TuckerLinear,
)


def _batched_mode_product(matrix_batch, tensor_batch, mode):
    moved = tensor_batch.movedim(mode + 1, 1)
    moved_shape = moved.shape
    multiplied = torch.bmm(matrix_batch, moved.flatten(2))
    return multiplied.reshape(moved_shape).movedim(1, mode + 1)


def _batched_qr_with_transport(factor_batch, tangent_batch):
    work_dtype = (
        torch.float64 if factor_batch.dtype == torch.float64 else torch.float32
    )
    work_factor = factor_batch.to(dtype=work_dtype)
    q_batch, r_batch = torch.linalg.qr(work_factor, mode="reduced")
    signs = torch.sign(torch.diagonal(r_batch, dim1=-2, dim2=-1))
    signs[signs == 0] = 1.0
    q_batch = q_batch * signs.unsqueeze(-2)
    r_batch = r_batch * signs.unsqueeze(-1)

    direction = tangent_batch.to(dtype=work_dtype)
    qt_direction = torch.bmm(q_batch.mT, direction)
    quotient = torch.linalg.solve_triangular(
        r_batch.mT,
        qt_direction.mT,
        upper=False,
    ).mT
    strictly_lower = torch.tril(quotient, diagonal=-1)
    omega = strictly_lower - strictly_lower.mT
    normal = direction - torch.bmm(q_batch, qt_direction)
    normal_times_r_inv = torch.linalg.solve_triangular(
        r_batch.mT,
        normal.mT,
        upper=False,
    ).mT
    transported = normal_times_r_inv + torch.bmm(q_batch, omega)
    d_r_batch = torch.triu(qt_direction - torch.bmm(omega, r_batch))
    return (
        q_batch.to(dtype=factor_batch.dtype),
        r_batch.to(dtype=factor_batch.dtype),
        transported.to(dtype=tangent_batch.dtype),
        d_r_batch.to(dtype=tangent_batch.dtype),
    )


@torch.no_grad()
def grouped_retract_tucker_modules_(
    model,
    *,
    optimizer=None,
    transport_optimizer_state: bool = False,
    compute_diagnostics: bool = False,
):
    """Retract Tucker modules with batched QR and batched core products."""
    if transport_optimizer_state and optimizer is None:
        raise ValueError("Vector transport requires the optimizer instance.")

    modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]
    groups = defaultdict(list)
    factor_groups = defaultdict(list)
    for module in modules:
        groups[
            (
                tuple(module.modes),
                tuple(module.ranks),
                tuple(module.active_factor_names),
            )
        ].append(module)
        for factor_name in module.active_factor_names:
            factor = getattr(module, factor_name)
            factor_groups[(tuple(factor.shape), factor.dtype, factor.device)].append(
                factor
            )

    transported_cores = 0
    transported_factors = 0
    r_by_factor = {}
    d_r_by_factor = {}
    for parameters in factor_groups.values():
        factor_batch = torch.stack(parameters)
        if transport_optimizer_state:
            factor_states = [optimizer.state.get(factor, {}) for factor in parameters]
            factor_momenta = [state.get("momentum_buffer") for state in factor_states]
            if any(momentum is None for momentum in factor_momenta):
                raise RuntimeError(
                    "Tucker vector transport requires momentum_buffer state for "
                    "every grouped factor."
                )
            q_batch, r_batch, transported_batch, d_r_batch = (
                _batched_qr_with_transport(
                    factor_batch,
                    torch.stack(factor_momenta),
                )
            )
            torch._foreach_copy_(
                [state["momentum_buffer"] for state in factor_states],
                list(transported_batch.unbind(0)),
            )
            d_r_by_factor.update(zip(parameters, d_r_batch.unbind(0)))
            transported_factors += len(parameters)
        else:
            q_batch, r_batch = torch.linalg.qr(factor_batch, mode="reduced")
            signs = torch.sign(torch.diagonal(r_batch, dim1=-2, dim2=-1))
            signs[signs == 0] = 1.0
            q_batch = q_batch * signs.unsqueeze(-2)
            r_batch = r_batch * signs.unsqueeze(-1)
        torch._foreach_copy_(parameters, list(q_batch.unbind(0)))
        r_by_factor.update(zip(parameters, r_batch.unbind(0)))

    for (_, ranks, active_factor_names), same_rank_modules in groups.items():
        r1, r2, r3, r4 = ranks
        cores = torch.stack(
            [
                module.core_matrix.reshape(r3, r4, r1, r2).permute(2, 3, 0, 1)
                for module in same_rank_modules
            ]
        )
        core_parameters = [module.core_matrix for module in same_rank_modules]
        core_states = None
        core_directions = None
        if transport_optimizer_state:
            core_states = [optimizer.state.get(core, {}) for core in core_parameters]
            core_momenta = [state.get("momentum_buffer") for state in core_states]
            if any(momentum is None for momentum in core_momenta):
                raise RuntimeError(
                    "Tucker vector transport requires momentum_buffer state for "
                    "every grouped core."
                )
            core_directions = torch.stack(
                [
                    momentum.reshape(r3, r4, r1, r2).permute(2, 3, 0, 1)
                    for momentum in core_momenta
                ]
            )
        for mode, factor_name in (
            (index, name)
            for index, name in enumerate(("U1", "U2", "U3", "U4"))
            if name in active_factor_names
        ):
            parameters = [getattr(module, factor_name) for module in same_rank_modules]
            r_batch = torch.stack([r_by_factor[factor] for factor in parameters])

            old_cores = cores
            cores = _batched_mode_product(r_batch, old_cores, mode)
            if transport_optimizer_state:
                d_r_batch = torch.stack(
                    [d_r_by_factor[factor] for factor in parameters]
                )
                core_directions = _batched_mode_product(
                    r_batch, core_directions, mode
                ) + _batched_mode_product(d_r_batch, old_cores, mode)

        core_matrices = [
            core.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
            for core in cores.unbind(0)
        ]
        torch._foreach_copy_(
            core_parameters, core_matrices
        )
        if transport_optimizer_state:
            core_direction_matrices = [
                direction.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
                for direction in core_directions.unbind(0)
            ]
            torch._foreach_copy_(
                [state["momentum_buffer"] for state in core_states],
                core_direction_matrices,
            )
            transported_cores += len(same_rank_modules)

    result = {
        "modules": len(modules),
        "factors": sum(len(module.active_factor_names) for module in modules),
        "transported_cores": transported_cores,
        "transported_factors": transported_factors,
    }
    if compute_diagnostics and modules:
        errors = torch.stack(
            [module.max_factor_orthogonality_error() for module in modules]
        )
        result["max_orthogonality_error"] = float(errors.max().cpu())
        result["mean_orthogonality_error"] = float(errors.mean().cpu())
        if transport_optimizer_state:
            tangent_errors = torch.stack(
                [
                    module.max_factor_momentum_tangency_error(optimizer)
                    for module in modules
                ]
            )
            result["max_momentum_tangency_error"] = float(
                tangent_errors.max().cpu()
            )
            result["mean_momentum_tangency_error"] = float(
                tangent_errors.mean().cpu()
            )
    return result
