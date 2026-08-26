"""Configurable Tucker and block-term parameterisation for ``nn.Linear`` modules.

The optional equal-parameter mode represents a logical matrix as

    W_effective = W_tucker + W_sparse_residual.

The Tucker ranks are chosen below the dense parameter budget.  The sparse
trainable residual contains exactly the remaining number of scalars, so every
parameter participates in the forward pass and the model-wide parameter count
is unchanged. Disabling equal-parameter mode removes that residual completely,
so the selected Tucker ranks alone determine the count; a full-mode Tucker
layer may contain slightly more parameters than its dense matrix.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import math
from pathlib import Path
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


def balanced_factor_pair(value: int, multiple: int = 1) -> tuple[int, int]:
    """Return exact factors nearest sqrt(value), without padded dead features."""
    if value <= 0:
        raise ValueError(f"feature dimension must be positive, got {value}")
    if multiple <= 0:
        raise ValueError(f"factor multiple must be positive, got {multiple}")
    left = math.isqrt(value)
    while left and (
        value % left
        or left % multiple
        or (value // left) % multiple
    ):
        left -= 1
    if not left:
        raise ValueError(
            f"feature dimension {value} has no exact factor pair divisible by {multiple}"
        )
    return left, value // left


def _parameter_count(
    modes: tuple[int, int, int, int],
    ranks: tuple[int, int, int, int],
) -> int:
    return sum(mode * rank for mode, rank in zip(modes, ranks)) + math.prod(ranks)


@lru_cache(maxsize=None)
def auto_tucker_ranks(
    in_features: int,
    out_features: int,
    mode_multiple: int = 1,
) -> tuple[int, int, int, int]:
    """Choose the largest valid full-Tucker parameterisation under ``n*m``.

    The four mode ranks may differ.  This is both less redundant and closer to
    the dense parameter budget than forcing a single rank on unequal modes.
    """
    n1, n2 = balanced_factor_pair(in_features, mode_multiple)
    m1, m2 = balanced_factor_pair(out_features, mode_multiple)
    modes = (n1, n2, m1, m2)
    budget = in_features * out_features
    best_key = None
    best_ranks = None

    # For fixed r1/r2/r3 the parameter count is monotone in r4, so r4 can be
    # solved analytically instead of performing a fourth nested loop.
    for r1 in range(1, n1 + 1):
        for r2 in range(1, n2 + 1):
            fixed12 = n1 * r1 + n2 * r2
            product12 = r1 * r2
            for r3 in range(1, m1 + 1):
                remaining = budget - fixed12 - m1 * r3
                denominator = product12 * r3 + m2
                r4 = min(m2, remaining // denominator)
                if r4 < 1:
                    continue
                ranks = (r1, r2, r3, r4)
                count = _parameter_count(modes, ranks)
                fractions = tuple(rank / mode for rank, mode in zip(ranks, modes))
                # Prefer the closest budget match, then balanced rank fractions.
                key = (
                    count,
                    min(fractions),
                    sum(fractions),
                    math.prod(ranks),
                    ranks,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_ranks = ranks

    if best_ranks is None:
        raise ValueError(
            f"No Tucker ranks fit the dense budget for {in_features}->{out_features}."
        )
    return best_ranks


def parse_tucker_rank_spec(
    scalar_rank: str | int,
    mode_ranks: str | None = None,
) -> str | int | tuple[int, int, int, int]:
    """Parse ``auto``, one scalar rank, or four comma-separated mode ranks."""
    if mode_ranks:
        try:
            parsed = tuple(int(part.strip()) for part in mode_ranks.split(","))
        except ValueError as error:
            raise ValueError("--tucker-ranks must contain four positive integers") from error
        if len(parsed) != 4 or any(rank <= 0 for rank in parsed):
            raise ValueError("--tucker-ranks must contain four positive integers")
        return parsed

    value = str(scalar_rank).strip().lower()
    if value == "auto":
        return "auto"
    try:
        parsed_scalar = int(value)
    except ValueError as error:
        raise ValueError("--tucker-rank must be 'auto' or a positive integer") from error
    if parsed_scalar <= 0:
        raise ValueError("--tucker-rank must be 'auto' or a positive integer")
    return parsed_scalar


@torch.no_grad()
def tucker_retract_(
    core: torch.Tensor,
    factors: list[torch.Tensor],
) -> torch.Tensor:
    """Make factor columns orthonormal without changing the Tucker tensor."""
    if core.ndim != len(factors):
        raise ValueError(
            f"Tucker core has {core.ndim} modes, but {len(factors)} factors "
            "were provided."
        )

    for mode, factor in enumerate(factors):
        if factor.ndim != 2:
            raise ValueError(
                f"Tucker factor {mode} must be a matrix, got shape "
                f"{tuple(factor.shape)}."
            )
        if factor.shape[1] != core.shape[mode]:
            raise ValueError(
                f"Tucker factor {mode} has rank {factor.shape[1]}, but core "
                f"mode {mode} has size {core.shape[mode]}."
            )

        Q, R = torch.linalg.qr(factor, mode="reduced")
        signs = torch.sign(torch.diagonal(R))
        signs[signs == 0] = 1.0
        factors[mode] = Q * signs
        core = torch.tensordot(
            R * signs[:, None],
            core,
            dims=([1], [mode]),
        ).movedim(0, mode)
    return core


def _mode_product(matrix: torch.Tensor, tensor: torch.Tensor, mode: int) -> torch.Tensor:
    """Multiply ``tensor`` by ``matrix`` along one Tucker mode."""

    return torch.tensordot(
        matrix,
        tensor,
        dims=([1], [mode]),
    ).movedim(0, mode)


@torch.no_grad()
def qr_retract_with_transport(
    factor: torch.Tensor,
    tangent: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Return sign-fixed QR factors and the differential applied to ``tangent``.

    The transported factor vector is the differentiated QR retraction on the
    Stiefel manifold.  ``dR`` is returned as well because the normal component
    removed from the factor must be absorbed into the Tucker-core momentum.
    """

    if factor.ndim != 2 or factor.shape[0] < factor.shape[1]:
        raise ValueError(
            "QR transport requires a tall-or-square matrix, got "
            f"{tuple(factor.shape)}."
        )
    if tangent is not None and tangent.shape != factor.shape:
        raise ValueError(
            f"Factor/tangent shape mismatch: {tuple(factor.shape)} vs "
            f"{tuple(tangent.shape)}."
        )

    work_dtype = torch.float64 if factor.dtype == torch.float64 else torch.float32
    work_factor = factor.to(dtype=work_dtype)
    Q, R = torch.linalg.qr(work_factor, mode="reduced")
    signs = torch.sign(torch.diagonal(R))
    signs[signs == 0] = 1.0
    Q = Q * signs
    R = R * signs[:, None]

    if tangent is None:
        return Q.to(dtype=factor.dtype), R.to(dtype=factor.dtype), None, None

    direction = tangent.to(dtype=work_dtype)
    qt_direction = Q.mT @ direction
    # right_solve(B, R) = B @ R^{-1}, expressed as a triangular solve.
    quotient = torch.linalg.solve_triangular(
        R.mT,
        qt_direction.mT,
        upper=False,
    ).mT
    strictly_lower = torch.tril(quotient, diagonal=-1)
    omega = strictly_lower - strictly_lower.mT

    normal = direction - Q @ qt_direction
    normal_times_r_inv = torch.linalg.solve_triangular(
        R.mT,
        normal.mT,
        upper=False,
    ).mT
    transported = normal_times_r_inv + Q @ omega
    dR = torch.triu(qt_direction - omega @ R)

    return (
        Q.to(dtype=factor.dtype),
        R.to(dtype=factor.dtype),
        transported.to(dtype=tangent.dtype),
        dR.to(dtype=tangent.dtype),
    )


@torch.no_grad()
def tucker_retract_with_transport_(
    core: torch.Tensor,
    factors: list[torch.Tensor],
    core_tangent: torch.Tensor | None,
    factor_tangents: list[torch.Tensor | None],
) -> tuple[
    torch.Tensor,
    list[torch.Tensor],
    torch.Tensor | None,
    list[torch.Tensor | None],
]:
    """Gauge-fix Tucker parameters and push first-order state through the map."""

    if len(factors) != core.ndim or len(factor_tangents) != len(factors):
        raise ValueError("Core, factors, and factor tangents must have matching order.")

    transported_factors: list[torch.Tensor] = []
    transported_tangents: list[torch.Tensor | None] = []
    for mode, (factor, factor_tangent) in enumerate(
        zip(factors, factor_tangents)
    ):
        Q, R, dQ, dR = qr_retract_with_transport(factor, factor_tangent)
        old_core = core
        core = _mode_product(R, old_core, mode)

        new_core_tangent = None
        if core_tangent is not None:
            new_core_tangent = _mode_product(R, core_tangent, mode)
        if dR is not None:
            factor_core_term = _mode_product(dR, old_core, mode)
            new_core_tangent = (
                factor_core_term
                if new_core_tangent is None
                else new_core_tangent + factor_core_term
            )
        core_tangent = new_core_tangent
        transported_factors.append(Q)
        transported_tangents.append(dQ)

    return core, transported_factors, core_tangent, transported_tangents


class TuckerLinear(nn.Module):
    """A full Tucker linear map with an optional exact-budget sparse residual."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: str | int | tuple[int, int, int, int] = "auto",
        bias: bool = True,
        equal_params: bool = True,
        init_std: float = 0.02,
        forward_mode: str = "auto",
        expected_tokens_per_forward: int = 1,
        mode_multiple: int = 1,
        extra_parameters: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.in_modes = balanced_factor_pair(self.in_features, mode_multiple)
        self.out_modes = balanced_factor_pair(self.out_features, mode_multiple)
        self.modes = (*self.in_modes, *self.out_modes)
        self.equal_params = bool(equal_params)
        self.init_std = float(init_std)
        if extra_parameters < 0:
            raise ValueError("extra_parameters must be non-negative")
        self.extra_parameters = int(extra_parameters)
        if forward_mode not in ("auto", "contract", "materialize"):
            raise ValueError(
                "Tucker forward_mode must be auto, contract, or materialize"
            )
        if expected_tokens_per_forward <= 0:
            raise ValueError("expected_tokens_per_forward must be positive")
        self.forward_mode = forward_mode
        self.expected_tokens_per_forward = int(expected_tokens_per_forward)

        if rank == "auto":
            ranks = auto_tucker_ranks(
                self.in_features,
                self.out_features,
                mode_multiple,
            )
            self.rank_policy = "auto"
        elif isinstance(rank, int):
            if rank <= 0:
                raise ValueError("Tucker scalar rank must be positive")
            ranks = tuple(min(rank, mode) for mode in self.modes)
            self.rank_policy = str(rank)
        else:
            if len(rank) != 4 or any(value <= 0 for value in rank):
                raise ValueError("Tucker mode ranks must be four positive integers")
            ranks = tuple(int(value) for value in rank)
            self.rank_policy = ",".join(str(value) for value in ranks)

        if any(rank_value > mode for rank_value, mode in zip(ranks, self.modes)):
            raise ValueError(
                f"Tucker ranks {ranks} exceed feature modes {self.modes} for "
                f"{self.in_features}->{self.out_features}."
            )
        self.ranks = ranks
        r1, r2, r3, r4 = ranks
        n1, n2 = self.in_modes
        m1, m2 = self.out_modes
        factory_kwargs = {"device": device, "dtype": dtype}

        self.tucker_parameter_count = _parameter_count(self.modes, self.ranks)
        self.dense_parameter_count = self.in_features * self.out_features
        if self.equal_params and self.tucker_parameter_count > self.dense_parameter_count:
            raise ValueError(
                f"Tucker ranks {self.ranks} require {self.tucker_parameter_count:,} "
                f"weight parameters, exceeding dense budget "
                f"{self.dense_parameter_count:,} for "
                f"{self.in_features}->{self.out_features}. Exact-budget mode "
                "requires rank='auto' or a smaller manual rank; pure Tucker "
                "mode (--no-tucker-equal-params) may use over-dense ranks."
            )

        if self.forward_mode == "auto":
            r1, r2, r3, r4 = self.ranks
            n1, n2 = self.in_modes
            m1, _ = self.out_modes
            contract_peak = self.expected_tokens_per_forward * max(
                r1 * n2,
                r1 * r2,
                r3 * r4,
                m1 * r4,
                self.out_features,
            )
            materialize_peak = max(
                math.prod(self.ranks),
                r3 * r4 * n1 * r2,
                r3 * r4 * self.in_features,
                m1 * r4 * self.in_features,
                self.dense_parameter_count,
            )
            # Materialising W avoids retaining token-wise Tucker intermediates
            # that are enormous for full-sequence training and the lm_head.
            self.resolved_forward_mode = (
                "materialize"
                if contract_peak > 4 * materialize_peak
                else "contract"
            )
        else:
            self.resolved_forward_mode = self.forward_mode

        residual_count = (
            self.dense_parameter_count
            + self.extra_parameters
            - self.tucker_parameter_count
            if self.equal_params
            else 0
        )
        if residual_count > self.dense_parameter_count:
            raise ValueError(
                f"Requested residual budget {residual_count:,} exceeds the "
                f"{self.dense_parameter_count:,} unique entries of the logical "
                "weight matrix."
            )
        self.residual_parameter_count = residual_count

        # Allocate only after all rank/budget checks so a bad manual rank
        # cannot OOM before producing its intended validation error.
        self.U1 = nn.Parameter(torch.empty(n1, r1, **factory_kwargs))
        self.U2 = nn.Parameter(torch.empty(n2, r2, **factory_kwargs))
        self.U3 = nn.Parameter(torch.empty(m1, r3, **factory_kwargs))
        self.U4 = nn.Parameter(torch.empty(m2, r4, **factory_kwargs))
        # Keeping the core two-dimensional makes its Muon role unambiguous.
        self.core_matrix = nn.Parameter(
            torch.empty(r3 * r4, r1 * r2, **factory_kwargs)
        )

        if residual_count:
            full_columns, partial_rows = divmod(
                residual_count, self.out_features
            )
            if full_columns:
                self.residual_matrix = nn.Parameter(
                    torch.empty(
                        self.out_features, full_columns, **factory_kwargs
                    )
                )
            else:
                self.register_parameter("residual_matrix", None)
            if partial_rows:
                self.residual_tail = nn.Parameter(
                    torch.empty(partial_rows, **factory_kwargs)
                )
            else:
                self.register_parameter("residual_tail", None)
            num_columns = full_columns + int(partial_rows > 0)
            # Spread the correction across the logical input dimension instead
            # of concentrating it in the first contiguous columns.
            residual_columns = (
                torch.arange(num_columns, device=device, dtype=torch.long)
                * self.in_features
                // num_columns
            )
            self.register_buffer(
                "residual_columns", residual_columns, persistent=False
            )
            self._residual_full_columns = full_columns
            self._residual_partial_rows = partial_rows
        else:
            self.register_parameter("residual_matrix", None)
            self.register_parameter("residual_tail", None)
            self.register_buffer(
                "residual_columns",
                torch.empty(0, device=device, dtype=torch.long),
                persistent=False,
            )
            self._residual_full_columns = 0
            self._residual_partial_rows = 0

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters(init_std=self.init_std)

        if self.equal_params:
            expected = self.dense_parameter_count + self.extra_parameters + (
                self.out_features if self.bias is not None else 0
            )
            actual = sum(parameter.numel() for parameter in self.parameters())
            if actual != expected:
                raise AssertionError(
                    f"Tucker exact-budget invariant failed: {actual} != {expected}"
                )

    def reset_parameters(self, *, init_std: float | None = None) -> None:
        target_std = self.init_std if init_std is None else float(init_std)
        for factor in (self.U1, self.U2, self.U3, self.U4):
            if factor.dtype in (torch.float16, torch.bfloat16):
                work = torch.empty_like(factor, dtype=torch.float32)
                nn.init.orthogonal_(work)
                factor.data.copy_(work)
            else:
                nn.init.orthogonal_(factor)
        core_elements = math.prod(self.ranks)
        core_std = target_std * math.sqrt(
            self.in_features * self.out_features / core_elements
        )
        nn.init.normal_(self.core_matrix, mean=0.0, std=core_std)
        if self.residual_matrix is not None:
            # Zero starts from a pure Tucker map; every residual scalar receives
            # a gradient on the first backward pass.
            nn.init.zeros_(self.residual_matrix)
        if self.residual_tail is not None:
            nn.init.zeros_(self.residual_tail)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight_parameter_count(self) -> int:
        return self.tucker_parameter_count + self.residual_parameter_count

    @property
    def residual_density(self) -> float:
        return self.residual_parameter_count / self.dense_parameter_count

    @property
    def forward_flops_per_token(self) -> float:
        """Configured forward-path FLOPs per token (two FLOPs per MAC)."""
        n1, n2 = self.in_modes
        m1, _ = self.out_modes
        r1, r2, r3, r4 = self.ranks
        if self.resolved_forward_mode == "materialize":
            reconstruction_macs = (
                r3 * r4 * n1 * r2 * r1
                + r3 * r4 * self.in_features * r2
                + m1 * r4 * self.in_features * r3
                + self.dense_parameter_count * r4
            )
            dense_macs = self.dense_parameter_count
            if hasattr(self, "_block_term_index"):
                # A BTD wrapper materializes all terms before one shared matmul.
                dense_macs = (
                    self.dense_parameter_count
                    if self._block_term_index == 0
                    else 0
                )
            return (
                2 * dense_macs
                + (
                    2 * reconstruction_macs
                    + self.residual_parameter_count
                )
                / self.expected_tokens_per_forward
            )
        macs = (
            self.in_features * r1
            + r1 * n2 * r2
            + r1 * r2 * r3 * r4
            + m1 * r3 * r4
            + self.out_features * r4
            + self.residual_parameter_count
        )
        return 2 * macs

    def _tucker_forward(self, x: torch.Tensor) -> torch.Tensor:
        n1, n2 = self.in_modes
        m1, m2 = self.out_modes
        r1, r2, r3, r4 = self.ranks
        shaped = x.reshape(*x.shape[:-1], n1, n2)
        hidden = torch.einsum("...ij,ia->...aj", shaped, self.U1)
        hidden = torch.einsum("...aj,jb->...ab", hidden, self.U2)
        hidden = F.linear(hidden.reshape(*x.shape[:-1], r1 * r2), self.core_matrix)
        hidden = hidden.reshape(*x.shape[:-1], r3, r4)
        hidden = torch.einsum("...cd,pc->...pd", hidden, self.U3)
        output = torch.einsum("...pd,qd->...pq", hidden, self.U4)
        return output.reshape(*x.shape[:-1], m1 * m2)

    def _residual_forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.residual_matrix is None and self.residual_tail is None:
            return None
        full_columns = self._residual_full_columns
        partial_rows = self._residual_partial_rows
        selected = x.index_select(-1, self.residual_columns)

        if full_columns:
            residual = F.linear(
                selected[..., :full_columns], self.residual_matrix
            )
        else:
            residual = x.new_zeros((*x.shape[:-1], self.out_features))

        if partial_rows:
            tail = (
                selected[..., full_columns].unsqueeze(-1)
                * self.residual_tail
            )
            # Clone before slice assignment so autograd keeps both branches.
            residual = residual.clone()
            residual[..., :partial_rows] = (
                residual[..., :partial_rows] + tail
            )
        return residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
            )
        if self.resolved_forward_mode == "materialize":
            weight = self.materialize_weight(dtype=self.core_matrix.dtype)
            return F.linear(x, weight, self.bias)

        output = self._tucker_forward(x)
        residual = self._residual_forward(x)
        if residual is not None:
            output = output + residual
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype)
        return output

    def materialize_weight(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        max_elements: int | None = None,
    ) -> torch.Tensor:
        """Return the logical ``[out_features, in_features]`` effective matrix."""
        elements = self.in_features * self.out_features
        if max_elements is not None and elements > max_elements:
            raise ValueError(
                f"Effective Tucker weight has {elements:,} elements, above "
                f"the configured metric limit {max_elements:,}."
            )
        U1 = self.U1.to(dtype=dtype)
        U2 = self.U2.to(dtype=dtype)
        U3 = self.U3.to(dtype=dtype)
        U4 = self.U4.to(dtype=dtype)
        core = self.core_matrix.to(dtype=dtype)
        r1, r2, r3, r4 = self.ranks
        n1, n2 = self.in_modes
        m1, m2 = self.out_modes
        core = core.reshape(r3, r4, r1, r2)
        weight = torch.einsum("cdab,ia->cdib", core, U1)
        weight = torch.einsum("cdib,jb->cdij", weight, U2)
        weight = torch.einsum("cdij,pc->pdij", weight, U3)
        weight = torch.einsum("pdij,qd->pqij", weight, U4).reshape(
            m1 * m2, n1 * n2
        )

        if self.residual_matrix is not None or self.residual_tail is not None:
            full_columns = self._residual_full_columns
            partial_rows = self._residual_partial_rows
            if full_columns:
                weight = weight.index_add(
                    1,
                    self.residual_columns[:full_columns],
                    self.residual_matrix.to(dtype=weight.dtype),
                )
            if partial_rows:
                tail_column = self.residual_columns[full_columns].reshape(1)
                tail = F.pad(
                    self.residual_tail.to(dtype=weight.dtype),
                    (0, self.out_features - partial_rows),
                ).unsqueeze(1)
                weight = weight.index_add(1, tail_column, tail)
        return weight

    @torch.no_grad()
    def retract_(self) -> None:
        """Orthonormalize all factors while preserving the effective weight."""
        r1, r2, r3, r4 = self.ranks
        factors = [self.U1, self.U2, self.U3, self.U4]
        core = self.core_matrix.reshape(r3, r4, r1, r2).permute(2, 3, 0, 1)
        core = tucker_retract_(core, factors)

        for parameter, retracted in zip(
            (self.U1, self.U2, self.U3, self.U4),
            factors,
        ):
            parameter.copy_(retracted)
        self.core_matrix.copy_(
            core.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
        )

    @torch.no_grad()
    def retract_with_optimizer_state_(self, optimizer) -> dict[str, int]:
        """Retract parameters and transport their first-order momentum state."""

        factors = [self.U1, self.U2, self.U3, self.U4]
        core_state = optimizer.state.get(self.core_matrix, {})
        factor_states = [optimizer.state.get(factor, {}) for factor in factors]
        core_momentum = core_state.get("momentum_buffer")
        factor_momenta = [state.get("momentum_buffer") for state in factor_states]

        missing = int(core_momentum is None) + sum(
            momentum is None for momentum in factor_momenta
        )
        if missing:
            raise RuntimeError(
                "Tucker vector transport requires momentum_buffer state for "
                f"the core and all four factors; {missing} buffers are missing."
            )

        r1, r2, r3, r4 = self.ranks
        core = self.core_matrix.reshape(r3, r4, r1, r2).permute(2, 3, 0, 1)
        core_direction = core_momentum.reshape(r3, r4, r1, r2).permute(
            2, 3, 0, 1
        )
        (
            core,
            retracted_factors,
            core_direction,
            transported_factor_momenta,
        ) = tucker_retract_with_transport_(
            core,
            factors,
            core_direction,
            factor_momenta,
        )

        for parameter, retracted, state, transported in zip(
            factors,
            retracted_factors,
            factor_states,
            transported_factor_momenta,
        ):
            parameter.copy_(retracted)
            state["momentum_buffer"].copy_(transported)
        self.core_matrix.copy_(
            core.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
        )
        core_state["momentum_buffer"].copy_(
            core_direction.permute(2, 3, 0, 1).reshape(r3 * r4, r1 * r2)
        )
        return {"cores": 1, "factors": 4}

    @torch.no_grad()
    def max_factor_momentum_tangency_error(self, optimizer) -> torch.Tensor:
        """Return max relative error in Q.T M + M.T Q after transport."""

        errors = []
        for factor in (self.U1, self.U2, self.U3, self.U4):
            momentum = optimizer.state[factor]["momentum_buffer"]
            violation = factor.mT @ momentum + momentum.mT @ factor
            denominator = torch.linalg.vector_norm(momentum).clamp_min(1e-12)
            errors.append(torch.linalg.vector_norm(violation) / denominator)
        return torch.stack(errors).max()

    @torch.no_grad()
    def max_factor_orthogonality_error(self) -> torch.Tensor:
        """Return max relative Frobenius error of ``U.T @ U`` from identity."""
        errors = []
        for factor in (self.U1, self.U2, self.U3, self.U4):
            gram = factor.T @ factor
            identity = torch.eye(
                gram.shape[0],
                device=gram.device,
                dtype=gram.dtype,
            )
            errors.append(
                torch.linalg.vector_norm(gram - identity)
                / math.sqrt(gram.shape[0])
            )
        return torch.stack(errors).max()

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"modes={self.modes}, ranks={self.ranks}, "
            f"equal_params={self.equal_params}, "
            f"extra_parameters={self.extra_parameters}, "
            f"forward_mode={self.resolved_forward_mode!r}, "
            f"residual={self.residual_parameter_count:,}, "
            f"bias={self.bias is not None}"
        )


class BlockTermTuckerLinear(nn.Module):
    """A linear map represented by a sum of independent Tucker terms."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: str | int | tuple[int, int, int, int],
        terms: int,
        bias: bool = True,
        init_std: float = 0.02,
        forward_mode: str = "auto",
        expected_tokens_per_forward: int = 1,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if terms <= 1:
            raise ValueError("Block-term Tucker requires at least two terms")
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_terms = int(terms)
        component_std = float(init_std) / math.sqrt(self.num_terms)
        self.components = nn.ModuleList(
            [
                TuckerLinear(
                    in_features,
                    out_features,
                    rank=rank,
                    bias=False,
                    equal_params=False,
                    init_std=component_std,
                    forward_mode=forward_mode,
                    expected_tokens_per_forward=expected_tokens_per_forward,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(self.num_terms)
            ]
        )
        first = self.components[0]
        self.in_modes = first.in_modes
        self.out_modes = first.out_modes
        self.modes = first.modes
        self.ranks = first.ranks
        self.rank_policy = first.rank_policy
        self.resolved_forward_mode = first.resolved_forward_mode
        self.equal_params = False
        self.extra_parameters = 0
        self.dense_parameter_count = first.dense_parameter_count
        self.tucker_parameter_count = sum(
            component.tucker_parameter_count for component in self.components
        )
        self.residual_parameter_count = 0
        self.register_parameter("residual_matrix", None)
        self.register_parameter("residual_tail", None)
        for index, component in enumerate(self.components):
            component._block_term_index = index

        if bias:
            self.bias = nn.Parameter(
                torch.zeros(out_features, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("bias", None)

    @property
    def weight_parameter_count(self) -> int:
        return self.tucker_parameter_count

    @property
    def residual_density(self) -> float:
        return 0.0

    @property
    def forward_flops_per_token(self) -> float:
        return sum(
            component.forward_flops_per_token for component in self.components
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"Expected last dimension {self.in_features}, got {x.shape[-1]}"
        )
        if self.resolved_forward_mode == "materialize":
            weight = self.materialize_weight(
                dtype=self.components[0].core_matrix.dtype
            )
            return F.linear(x, weight, self.bias)

        output = self.components[0]._tucker_forward(x)
        for component in self.components[1:]:
            output = output + component._tucker_forward(x)
        if self.bias is not None:
            output = output + self.bias.to(dtype=output.dtype)
        return output

    def materialize_weight(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        max_elements: int | None = None,
    ) -> torch.Tensor:
        weight = self.components[0].materialize_weight(
            dtype=dtype,
            max_elements=max_elements,
        )
        for component in self.components[1:]:
            weight = weight + component.materialize_weight(
                dtype=dtype,
                max_elements=max_elements,
            )
        return weight

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"terms={self.num_terms}, modes={self.modes}, ranks={self.ranks}, "
            f"forward_mode={self.resolved_forward_mode!r}, "
            f"bias={self.bias is not None}"
        )


@dataclass(frozen=True)
class TuckerReplacementStats:
    modules: int
    terms_per_module: int
    parameters_before: int
    parameters_after: int
    tucker_parameters: int
    residual_parameters: int
    residual_matrix_parameters: int
    residual_tail_parameters: int
    dense_equivalent_parameters: int
    target_parameter_count: int
    target_parameter_tolerance: int
    parameter_difference_from_target: int | None
    plans: tuple[
        tuple[tuple[int, int], tuple[int, int, int, int], int, int], ...
    ]
    forward_modes: tuple[tuple[str, int], ...]


@torch.no_grad()
def retract_tucker_modules_(
    model: nn.Module,
    *,
    optimizer=None,
    transport_optimizer_state: bool = False,
    compute_diagnostics: bool = False,
) -> dict[str, float | int]:
    """Retract every Tucker layer after an optimizer step."""
    modules = [
        module for module in model.modules() if isinstance(module, TuckerLinear)
    ]
    transported_cores = 0
    transported_factors = 0
    for module in modules:
        if transport_optimizer_state:
            if optimizer is None:
                raise ValueError("Vector transport requires the optimizer instance.")
            transported = module.retract_with_optimizer_state_(optimizer)
            transported_cores += transported["cores"]
            transported_factors += transported["factors"]
        else:
            module.retract_()

    result: dict[str, float | int] = {
        "modules": len(modules),
        "factors": 4 * len(modules),
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


def replace_all_linears_with_tucker(model: nn.Module, config) -> TuckerReplacementStats:
    """Recursively replace every independent ``nn.Linear`` in a Llama model."""
    if getattr(config, "model", "llama") != "llama":
        raise ValueError(
            "All-Linear Tucker mode currently requires --model llama. GPTBase "
            "ties lm_head.weight to the token embedding, so replacing that head "
            "would silently change the model's weight sharing."
        )
    if getattr(config, "fp8", False) or getattr(config, "fp8_optim", False):
        raise ValueError(
            "Tucker Linear parameterisation is not compatible with --fp8 "
            "or --fp8-optim yet."
        )

    rank_spec = parse_tucker_rank_spec(
        getattr(config, "tucker_rank", "auto"),
        getattr(config, "tucker_ranks", None),
    )
    attention_rank_spec = (
        parse_tucker_rank_spec(
            "auto", getattr(config, "tucker_attention_ranks", None)
        )
        if getattr(config, "tucker_attention_ranks", None)
        else None
    )
    gate_up_rank_spec = (
        parse_tucker_rank_spec(
            "auto", getattr(config, "tucker_gate_up_ranks", None)
        )
        if getattr(config, "tucker_gate_up_ranks", None)
        else None
    )
    down_rank_spec = (
        parse_tucker_rank_spec(
            "auto", getattr(config, "tucker_down_ranks", None)
        )
        if getattr(config, "tucker_down_ranks", None)
        else None
    )
    rank_plan_source = getattr(config, "tucker_rank_plan", None)
    rank_plan: dict[str, tuple[int, int, int, int]] | None = None
    if rank_plan_source:
        if isinstance(rank_plan_source, dict):
            payload = rank_plan_source
        else:
            with Path(rank_plan_source).open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        raw_plan = payload.get("module_ranks", payload)
        if not isinstance(raw_plan, dict):
            raise ValueError("--tucker-rank-plan must contain a JSON object")
        rank_plan = {}
        for module_name, values in raw_plan.items():
            if not isinstance(values, (list, tuple)) or len(values) != 4:
                raise ValueError(
                    f"Adaptive Tucker ranks for {module_name!r} must contain "
                    "four integers"
                )
            ranks = tuple(int(value) for value in values)
            if any(value <= 0 for value in ranks):
                raise ValueError(
                    f"Adaptive Tucker ranks for {module_name!r} must be positive"
                )
            rank_plan[str(module_name)] = ranks

    def rank_spec_for_layer(full_name: str):
        if rank_plan is not None:
            try:
                return rank_plan[full_name]
            except KeyError as error:
                raise ValueError(
                    f"Adaptive Tucker rank plan has no entry for {full_name!r}"
                ) from error
        if attention_rank_spec is not None and full_name.endswith(
            ("q_proj", "k_proj", "v_proj", "o_proj")
        ):
            return attention_rank_spec
        if gate_up_rank_spec is not None and full_name.endswith(
            ("gate_proj", "up_proj")
        ):
            return gate_up_rank_spec
        if down_rank_spec is not None and full_name.endswith("down_proj"):
            return down_rank_spec
        return rank_spec

    rank_policy_summary = {
        "default": rank_spec,
        "attention": attention_rank_spec,
        "gate_up": gate_up_rank_spec,
        "down": down_rank_spec,
    }
    tucker_terms = int(getattr(config, "tucker_terms", 1))
    if tucker_terms <= 0:
        raise ValueError("--tucker-terms must be positive")
    equal_params = bool(getattr(config, "tucker_equal_params", True))
    if tucker_terms > 1 and equal_params:
        raise ValueError(
            "Block-term Tucker requires --no-tucker-equal-params"
        )
    forward_mode = getattr(config, "tucker_forward_mode", "auto")
    mode_multiple = int(getattr(config, "tucker_mode_multiple", 1))
    expected_tokens_per_forward = (
        int(getattr(config, "batch_size", 1)) * int(config.sequence_length)
    )
    keep_adamw_matrices_dense = bool(
        getattr(config, "tucker_dense_adamw_matrices", False)
    )

    def keep_dense(full_name: str) -> bool:
        return keep_adamw_matrices_dense and full_name == "lm_head"

    parameters_before = sum(parameter.numel() for parameter in model.parameters())
    target_parameter_count = int(getattr(config, "target_parameter_count", 0))
    target_parameter_tolerance = int(
        getattr(config, "target_parameter_tolerance", 0)
    )
    if target_parameter_count < 0:
        raise ValueError("--target-parameter-count must be non-negative")
    if target_parameter_tolerance < 0:
        raise ValueError("--target-parameter-tolerance must be non-negative")

    # In exact-budget mode the target actively determines the trainable
    # correction budget. In pure Tucker mode it is validation-only: ranks are
    # the sole source of parameters and no filler/correction is created.
    expected_parameters_after = target_parameter_count or parameters_before
    if equal_params and expected_parameters_after < parameters_before:
        raise ValueError(
            f"--target-parameter-count={expected_parameters_after:,} is below "
            f"the same model's dense-Linear count {parameters_before:,}. Use "
            "--no-tucker-equal-params for a rank-only Tucker model instead."
        )

    linear_names: list[str] = []

    def collect_linear_names(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in parent.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if not keep_dense(full_name):
                    linear_names.append(full_name)
            else:
                collect_linear_names(child, full_name)

    collect_linear_names(model)
    if rank_plan is not None:
        expected_names = set(linear_names)
        supplied_names = set(rank_plan)
        missing = sorted(expected_names - supplied_names)
        extra = sorted(supplied_names - expected_names)
        if missing or extra:
            raise ValueError(
                "Adaptive Tucker rank plan does not match model modules: "
                f"missing={missing}, extra={extra}"
            )
    target_extras = {name: 0 for name in linear_names}
    parameter_gap = (
        expected_parameters_after - parameters_before if equal_params else 0
    )
    if parameter_gap:
        # Spread a small control-model gap over attention output projections.
        # For the supplied rank-1023 tensorized model this is exactly 1,026
        # parameters per block, matching the original standard-attention total.
        candidates = [name for name in linear_names if name.endswith("o_proj")]
        if not candidates:
            candidates = [name for name in linear_names if name != "lm_head"]
        quotient, remainder = divmod(parameter_gap, len(candidates))
        for index, name in enumerate(candidates):
            target_extras[name] = quotient + int(index < remainder)

    modules: list[TuckerLinear | BlockTermTuckerLinear] = []
    plan_counts: dict[tuple[tuple[int, int], tuple[int, int, int, int], int], int] = {}

    def replace(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear):
                if keep_dense(full_name):
                    continue
                init_std = float(config.init_std)
                if full_name.endswith(("o_proj", "down_proj")):
                    init_std /= math.sqrt(2 * config.n_layer)
                if tucker_terms == 1:
                    replacement = TuckerLinear(
                        child.in_features,
                        child.out_features,
                        rank=rank_spec_for_layer(full_name),
                        bias=child.bias is not None,
                        equal_params=equal_params,
                        init_std=init_std,
                        forward_mode=forward_mode,
                        expected_tokens_per_forward=expected_tokens_per_forward,
                        mode_multiple=mode_multiple,
                        extra_parameters=target_extras[full_name],
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                else:
                    replacement = BlockTermTuckerLinear(
                        child.in_features,
                        child.out_features,
                        rank=rank_spec_for_layer(full_name),
                        terms=tucker_terms,
                        bias=child.bias is not None,
                        init_std=init_std,
                        forward_mode=forward_mode,
                        expected_tokens_per_forward=expected_tokens_per_forward,
                        device=child.weight.device,
                        dtype=child.weight.dtype,
                    )
                if child.bias is not None:
                    with torch.no_grad():
                        replacement.bias.copy_(child.bias)
                setattr(parent, child_name, replacement)
                modules.append(replacement)
                key = (
                    (replacement.in_features, replacement.out_features),
                    replacement.ranks,
                    replacement.residual_parameter_count,
                )
                plan_counts[key] = plan_counts.get(key, 0) + 1
            else:
                replace(child, full_name)

    replace(model)
    parameters_after = sum(parameter.numel() for parameter in model.parameters())
    if not modules:
        raise ValueError("Tucker mode was requested but the model has no nn.Linear modules.")
    if equal_params and expected_parameters_after != parameters_after:
        raise AssertionError(
            "All-Linear Tucker replacement missed the exact parameter target: "
            f"expected {expected_parameters_after:,}, got {parameters_after:,} "
            f"(dense-Linear model: {parameters_before:,})."
        )
    parameter_difference_from_target = (
        parameters_after - target_parameter_count
        if target_parameter_count
        else None
    )
    if (
        not equal_params
        and parameter_difference_from_target is not None
        and abs(parameter_difference_from_target) > target_parameter_tolerance
    ):
        raise ValueError(
            "Pure Tucker rank-only parameter check failed: "
            f"rank policy {rank_policy_summary!r} produced "
            f"{parameters_after:,} parameters, "
            f"target is {target_parameter_count:,} "
            f"(difference {parameter_difference_from_target:+,}, allowed "
            f"±{target_parameter_tolerance:,}). Change --tucker-rank/"
            "--tucker-ranks or increase --target-parameter-tolerance."
        )

    forward_mode_counts: dict[str, int] = {}
    for module in modules:
        mode = module.resolved_forward_mode
        forward_mode_counts[mode] = forward_mode_counts.get(mode, 0) + 1

    stats = TuckerReplacementStats(
        modules=len(modules),
        terms_per_module=tucker_terms,
        parameters_before=parameters_before,
        parameters_after=parameters_after,
        tucker_parameters=sum(module.tucker_parameter_count for module in modules),
        residual_parameters=sum(module.residual_parameter_count for module in modules),
        residual_matrix_parameters=sum(
            0 if module.residual_matrix is None else module.residual_matrix.numel()
            for module in modules
        ),
        residual_tail_parameters=sum(
            0 if module.residual_tail is None else module.residual_tail.numel()
            for module in modules
        ),
        dense_equivalent_parameters=sum(module.dense_parameter_count for module in modules),
        target_parameter_count=target_parameter_count,
        target_parameter_tolerance=target_parameter_tolerance,
        parameter_difference_from_target=parameter_difference_from_target,
        plans=tuple(
            (shape, ranks, residual, count)
            for (shape, ranks, residual), count in sorted(plan_counts.items())
        ),
        forward_modes=tuple(sorted(forward_mode_counts.items())),
    )
    model._tucker_replacement_stats = stats
    if (
        stats.residual_matrix_parameters + stats.residual_tail_parameters
        != stats.residual_parameters
    ):
        raise AssertionError("Tucker residual parameter split is inconsistent.")

    print("\nTucker Linear replacement:")
    print(f"  modules: {stats.modules}")
    print(f"  terms per module: {stats.terms_per_module}")
    print(
        f"  total parameters: {stats.parameters_before:,} -> "
        f"{stats.parameters_after:,}"
    )
    if stats.target_parameter_count:
        print(
            f"  validation target: {stats.target_parameter_count:,}; "
            f"difference={stats.parameter_difference_from_target:+,}; "
            f"tolerance=±{stats.target_parameter_tolerance:,}"
        )
    print(
        f"  Tucker/residual weight parameters: "
        f"{stats.tucker_parameters:,} / {stats.residual_parameters:,}"
    )
    print(
        f"  residual 2-D/1-D split: "
        f"{stats.residual_matrix_parameters:,} / "
        f"{stats.residual_tail_parameters:,}"
    )
    print(f"  resolved forward modes: {dict(stats.forward_modes)}")
    for (in_features, out_features), ranks, residual, count in stats.plans:
        print(
            f"  {count:>2}x {in_features}->{out_features}: "
            f"terms={stats.terms_per_module}, ranks={ranks}, "
            f"residual={residual:,} "
            f"({residual / (in_features * out_features):.2%})"
        )
    if equal_params and stats.residual_parameters:
        residual_density = (
            stats.residual_parameters / stats.dense_equivalent_parameters
        )
        if residual_density > 0.10:
            warnings.warn(
                f"The chosen manual Tucker rank leaves {residual_density:.1%} "
                "of Linear weights in the sparse residual. Use --tucker-rank auto "
                "for a Tucker-dominant exact-parameter experiment.",
                stacklevel=2,
            )
    return stats
