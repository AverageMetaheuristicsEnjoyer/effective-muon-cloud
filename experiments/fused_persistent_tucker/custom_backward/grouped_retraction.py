"""Batched QR gauge fixing for equal-shaped Tucker modules."""

from __future__ import annotations

from collections import defaultdict

import torch

from models.tucker_linear import (
    TuckerLinear,
    retract_tucker_modules_ as _reference_retract_tucker_modules,
)


@torch.no_grad()
def grouped_retract_tucker_modules_(
    model,
    *,
    optimizer=None,
    transport_optimizer_state: bool = False,
    compute_diagnostics: bool = False,
):
    """Retract equal-rank modules with batched QR and batched core products.

    Vector transport deliberately falls back to the production implementation;
    the selected training launcher does not enable it.  Batching changes only
    scheduling and preserves each module's independent Tucker parameters.
    """
    if transport_optimizer_state:
        return _reference_retract_tucker_modules(
            model,
            optimizer=optimizer,
            transport_optimizer_state=True,
            compute_diagnostics=compute_diagnostics,
        )

    modules = [module for module in model.modules() if isinstance(module, TuckerLinear)]
    groups = defaultdict(list)
    for module in modules:
        groups[(tuple(module.modes), tuple(module.ranks))].append(module)

    for (_, ranks), same_rank_modules in groups.items():
        r1, r2, r3, r4 = ranks
        cores = torch.stack(
            [
                module.core_matrix.reshape(r3, r4, r1, r2).permute(2, 3, 0, 1)
                for module in same_rank_modules
            ]
        )
        for mode, factor_name in enumerate(("U1", "U2", "U3", "U4")):
            parameters = [getattr(module, factor_name) for module in same_rank_modules]
            factor_batch = torch.stack(parameters)
            q_batch, r_batch = torch.linalg.qr(factor_batch, mode="reduced")
            signs = torch.sign(torch.diagonal(r_batch, dim1=-2, dim2=-1))
            signs[signs == 0] = 1.0
            q_batch = q_batch * signs.unsqueeze(-2)
            r_batch = r_batch * signs.unsqueeze(-1)
            torch._foreach_copy_(parameters, list(q_batch.unbind(0)))

            # Batch is dimension zero; move the selected Tucker mode next to it
            # and reduce the remaining modes to a single GEMM column dimension.
            moved = cores.movedim(mode + 1, 1)
            moved_shape = moved.shape
            multiplied = torch.bmm(r_batch, moved.flatten(2))
            cores = multiplied.reshape(moved_shape).movedim(1, mode + 1)

        core_matrices = [
            core.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
            for core in cores.unbind(0)
        ]
        torch._foreach_copy_(
            [module.core_matrix for module in same_rank_modules], core_matrices
        )

    result = {
        "modules": len(modules),
        "factors": 4 * len(modules),
        "transported_cores": 0,
        "transported_factors": 0,
    }
    if compute_diagnostics and modules:
        errors = torch.stack(
            [module.max_factor_orthogonality_error() for module in modules]
        )
        result["max_orthogonality_error"] = float(errors.max().cpu())
        result["mean_orthogonality_error"] = float(errors.mean().cpu())
    return result
