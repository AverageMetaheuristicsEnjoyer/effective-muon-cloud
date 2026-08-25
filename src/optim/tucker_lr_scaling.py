"""Matrix-free spectral helpers for coupled Tucker learning-rate scaling.

For the TuckerLinear parameterization used in this repository,

    W = (U3 kron U4) @ G @ (U1 kron U2).T,

and therefore

    ||W||_2 <= ||G||_2 * product_i ||Ui||_2.

For simultaneous candidate updates Fi' = Fi - alpha * Di and alpha <= 1,
the reconstructed-weight change is bounded by

    ||Delta W||_2
      <= product_i (sigma_i + alpha * rho_i) - product_i sigma_i
      <= alpha * (product_i (sigma_i + rho_i) - product_i sigma_i).

The helpers below never materialize W or Delta W in normal training.  The
optional strict debug check in TensorionOptimizer is restricted to tiny test
operators.
"""

from __future__ import annotations

from typing import MutableMapping, Sequence

import torch


def tucker_paper_mup_lr_multipliers(
    core: torch.Tensor,
    factors: Sequence[torch.Tensor],
) -> tuple[float, float, float, float, float]:
    """Return the static structure-aware LR multipliers from Qiu et al.

    The paper transfers a dense-layer Adam learning rate to each learnable
    dense component ``Gi`` as ``kappa_i = (d_in / d_in_i) / k``.  Our Tucker
    operator is evaluated as

        (U3 kron U4) @ G @ (U1 kron U2).T,

    so the five component input widths are ``r1*r2, n1, n2, r3, r4`` for
    ``G, U1.T, U2.T, U3, U4`` respectively.  The returned order matches
    ``(core, U1, U2, U3, U4)``.
    """

    if len(factors) != 4:
        raise ValueError("Paper-muP Tucker scaling expects four factors")
    if core.ndim != 2:
        raise ValueError(f"Tucker core must be a matrix, got {tuple(core.shape)}")
    if any(factor.ndim != 2 for factor in factors):
        raise ValueError("Every Tucker factor must be a matrix")

    U1, U2, U3, U4 = factors
    n1, r1 = (int(value) for value in U1.shape)
    n2, r2 = (int(value) for value in U2.shape)
    _, r3 = (int(value) for value in U3.shape)
    _, r4 = (int(value) for value in U4.shape)
    expected_core_shape = (r3 * r4, r1 * r2)
    if tuple(core.shape) != expected_core_shape:
        raise ValueError(
            f"Tucker core shape {tuple(core.shape)} does not match "
            f"factor ranks {expected_core_shape}"
        )

    dense_input_width = n1 * n2
    component_count = 5.0
    local_input_widths = (r1 * r2, n1, n2, r3, r4)
    return tuple(
        dense_input_width / (component_count * local_width)
        for local_width in local_input_widths
    )


def _initial_power_vector(size: int, *, device: torch.device) -> torch.Tensor:
    """Return a deterministic non-zero FP32 vector without consuming RNG state."""

    vector = torch.linspace(1.0, 2.0, size, device=device, dtype=torch.float32)
    return vector / torch.linalg.vector_norm(vector).clamp_min(1e-12)


@torch.no_grad()
def warm_started_spectral_norm(
    matrix: torch.Tensor,
    state: MutableMapping,
    *,
    prefix: str,
    power_iters: int = 1,
    eps: float = 1e-8,
    exact_svd_debug: bool = False,
) -> torch.Tensor:
    """Estimate ``||matrix||_2`` in FP32 and checkpoint the power vectors.

    ``state`` is an optimizer per-parameter state dictionary, so the left and
    right vectors are automatically included in optimizer checkpoints.
    """

    if matrix.ndim != 2:
        raise ValueError(f"Spectral norm expects a matrix, got {tuple(matrix.shape)}")
    if power_iters < 1:
        raise ValueError(f"power_iters must be >= 1, got {power_iters}")

    work = matrix.detach().to(dtype=torch.float32)
    if exact_svd_debug:
        return torch.linalg.matrix_norm(work, ord=2)

    rows, columns = work.shape
    right_key = f"{prefix}_right"
    left_key = f"{prefix}_left"
    right = state.get(right_key)
    if (
        right is None
        or right.shape != (columns,)
        or right.device != work.device
        or right.dtype != torch.float32
    ):
        right = _initial_power_vector(columns, device=work.device)
    else:
        right = right.detach()

    left = None
    for _ in range(power_iters):
        left = work @ right
        left = left / torch.linalg.vector_norm(left).clamp_min(eps)
        right_candidate = work.mT @ left
        right_norm = torch.linalg.vector_norm(right_candidate)
        right = torch.where(
            right_norm > eps,
            right_candidate / right_norm.clamp_min(eps),
            _initial_power_vector(columns, device=work.device),
        )

    if left is None:  # pragma: no cover - guarded by power_iters validation
        raise RuntimeError("Power iteration did not run")
    left = work @ right
    sigma = torch.linalg.vector_norm(left)
    left = left / sigma.clamp_min(eps)
    state[right_key] = right.detach().clone()
    state[left_key] = left.detach().clone()
    return sigma


def tucker_spectral_denominator(
    sigmas: Sequence[torch.Tensor | float],
    rhos: Sequence[torch.Tensor | float],
    *,
    mode: str,
) -> torch.Tensor:
    """Return the full-product or first-order Tucker update denominator."""

    if len(sigmas) != len(rhos) or not sigmas:
        raise ValueError("sigmas and rhos must be non-empty and have equal length")
    reference = next(
        (value for value in (*sigmas, *rhos) if isinstance(value, torch.Tensor)),
        None,
    )
    if reference is None:
        reference = torch.tensor(0.0, dtype=torch.float32)
    values_sigma = torch.stack(
        [torch.as_tensor(value, device=reference.device, dtype=torch.float32) for value in sigmas]
    )
    values_rho = torch.stack(
        [torch.as_tensor(value, device=reference.device, dtype=torch.float32) for value in rhos]
    )

    if mode == "spectron_bound":
        return torch.prod(values_sigma + values_rho) - torch.prod(values_sigma)
    if mode == "first_order":
        terms = []
        for index, rho in enumerate(values_rho):
            if len(values_sigma) == 1:
                others = values_sigma.new_tensor(1.0)
            else:
                others = torch.prod(
                    torch.cat((values_sigma[:index], values_sigma[index + 1 :]))
                )
            terms.append(rho * others)
        return torch.stack(terms).sum()
    raise ValueError(f"Unknown Tucker LR scaling mode: {mode!r}")


def two_factor_spectron_denominator(
    sigma_a: torch.Tensor | float,
    sigma_b: torch.Tensor | float,
    rho_a: torch.Tensor | float = 1.0,
    rho_b: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Convenience helper used to verify the Spectron two-factor reduction."""

    return tucker_spectral_denominator(
        [sigma_a, sigma_b],
        [rho_a, rho_b],
        mode="spectron_bound",
    )


__all__ = [
    "tucker_paper_mup_lr_multipliers",
    "tucker_spectral_denominator",
    "two_factor_spectron_denominator",
    "warm_started_spectral_norm",
]
