"""Tensorion optimizer from Bogachev et al. (2026).

Reference: "Tensorion: A Tensor-Aware Generalization of the Muon Optimizer",
Algorithm 1 and Eq. (24), https://arxiv.org/abs/2606.25975.

Tensorion applies a Muon-style orthogonalized momentum update to an offline,
shape-selected unfolding of every higher-order tensor parameter. Eligible
matrices use Muon and the remaining parameters use AdamW in one optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence
import warnings

import torch

from optim.tucker_lr_scaling import (
    tucker_paper_mup_lr_multipliers,
    tucker_spectral_denominator,
    warm_started_spectral_norm,
)


@dataclass(frozen=True)
class UnfoldingPlan:
    """A fixed per-parameter unfolding selected before optimization."""

    tensor_shape: tuple[int, ...]
    row_modes: tuple[int, ...]
    column_modes: tuple[int, ...]
    rows: int
    columns: int


@dataclass(frozen=True)
class TuckerCoupledSpec:
    """The five linked parameters forming one reconstructed Tucker weight."""

    name: str
    core: torch.nn.Parameter
    factors: tuple[
        torch.nn.Parameter,
        torch.nn.Parameter,
        torch.nn.Parameter,
        torch.nn.Parameter,
    ]


def select_balanced_unfolding(shape: Sequence[int]) -> UnfoldingPlan:
    """Select the offline Tensorion unfolding from Eq. (24).

    Complementary unfoldings differ only by transposition, so only one member
    of every ``tau ~ tau^c`` pair is considered.  The chosen unfolding
    maximizes ``min(prod(tau), prod(tau^c))`` and is therefore as close to
    square as the tensor modes permit.
    """

    tensor_shape = tuple(int(size) for size in shape)
    if len(tensor_shape) < 2:
        raise ValueError(
            f"Tensorion needs a tensor of order at least 2, got {tensor_shape}."
        )
    if any(size <= 0 for size in tensor_shape):
        raise ValueError(f"Tensorion dimensions must be positive, got {tensor_shape}.")

    order = len(tensor_shape)
    full_mask = (1 << order) - 1
    best_plan = None
    best_score = -1

    for mask in range(1, full_mask):
        complement = full_mask ^ mask
        if mask > complement:
            continue

        row_modes = tuple(index for index in range(order) if mask & (1 << index))
        column_modes = tuple(index for index in range(order) if not mask & (1 << index))
        rows = math.prod(tensor_shape[index] for index in row_modes)
        columns = math.prod(tensor_shape[index] for index in column_modes)

        # Keep the smaller side as rows.  This is equivalent up to transpose
        # and reduces the Newton--Schulz workspace.
        if rows > columns:
            row_modes, column_modes = column_modes, row_modes
            rows, columns = columns, rows

        score = min(rows, columns)
        if score > best_score:
            best_score = score
            best_plan = UnfoldingPlan(
                tensor_shape=tensor_shape,
                row_modes=row_modes,
                column_modes=column_modes,
                rows=rows,
                columns=columns,
            )

    if best_plan is None:  # pragma: no cover - guarded by the order check
        raise RuntimeError(f"Failed to select an unfolding for {tensor_shape}.")
    return best_plan


def unfold_tensor(tensor: torch.Tensor, plan: UnfoldingPlan) -> torch.Tensor:
    """Apply ``plan`` and return its matrix unfolding."""

    if tensor.numel() != math.prod(plan.tensor_shape):
        raise ValueError(
            f"Tensor with {tensor.numel():,} elements cannot use logical shape "
            f"{plan.tensor_shape}."
        )
    permutation = plan.row_modes + plan.column_modes
    return (
        tensor.reshape(plan.tensor_shape)
        .permute(permutation)
        .reshape(plan.rows, plan.columns)
    )


def fold_tensor(matrix: torch.Tensor, plan: UnfoldingPlan) -> torch.Tensor:
    """Invert :func:`unfold_tensor` and return ``plan.tensor_shape``."""

    if tuple(matrix.shape) != (plan.rows, plan.columns):
        raise ValueError(
            f"Expected unfolding {(plan.rows, plan.columns)}, got {tuple(matrix.shape)}."
        )
    permutation = plan.row_modes + plan.column_modes
    permuted_shape = tuple(plan.tensor_shape[index] for index in permutation)
    inverse = [0] * len(permutation)
    for position, original_mode in enumerate(permutation):
        inverse[original_mode] = position
    return matrix.reshape(permuted_shape).permute(inverse)


def _orthogonalize_svd(matrix: torch.Tensor) -> torch.Tensor:
    original_dtype = matrix.dtype
    work = matrix.float() if matrix.dtype in (torch.float16, torch.bfloat16) else matrix
    left, _, right_t = torch.linalg.svd(work, full_matrices=False)
    return (left @ right_t).to(dtype=original_dtype)


def _orthogonalize_newton_schulz(
    matrix: torch.Tensor,
    steps: int,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Muon's quintic Newton--Schulz approximation of ``U @ V.T``."""

    if steps < 1:
        raise ValueError(f"Tensorion Newton--Schulz steps must be >= 1, got {steps}.")
    original_dtype = matrix.dtype
    # CUDA supports Muon's low-precision matrix multiplies directly; retaining
    # BF16 there is important for practical memory use and throughput. CPU
    # linear algebra support is more limited, so CPU-only runs upcast.
    work = (
        matrix.float()
        if matrix.device.type == "cpu"
        and matrix.dtype in (torch.float16, torch.bfloat16)
        else matrix
    )
    transposed = work.shape[0] > work.shape[1]
    if transposed:
        work = work.mT
    work = work / (torch.linalg.vector_norm(work) + eps)

    # Coefficients used by the Muon implementation referenced by Algorithm 1.
    for _ in range(steps):
        gram = work @ work.mT
        work = 3.4445 * work - 4.7750 * (gram @ work) + 2.0315 * ((gram @ gram) @ work)

    if transposed:
        work = work.mT
    return work.to(dtype=original_dtype)


def tensorion_direction(
    momentum: torch.Tensor,
    plan: UnfoldingPlan,
    *,
    orthogonalization: str = "ns",
    ns_steps: int = 5,
) -> torch.Tensor:
    """Compute ``fold(NS(unfold(momentum, tau)), tau)`` from Algorithm 1."""

    matrix = unfold_tensor(momentum, plan)
    if orthogonalization == "ns":
        update = _orthogonalize_newton_schulz(matrix, ns_steps)
    elif orthogonalization == "svd":
        update = _orthogonalize_svd(matrix)
    else:
        raise ValueError(
            "Tensorion orthogonalization must be 'ns' or 'svd', got "
            f"{orthogonalization!r}."
        )
    return fold_tensor(update, plan).reshape_as(momentum)


def stiefel_tangent_projection(
    point: torch.Tensor,
    vector: torch.Tensor,
) -> torch.Tensor:
    """Project ``vector`` onto the Stiefel tangent space at ``point``.

    For a column-orthonormal factor ``X``, the Euclidean-metric projection is

    ``Proj_X(U) = U - X sym(X.T @ U)``.

    Tucker factors are small, but training stores them in BF16.  Computing the
    Gram term in FP32 materially improves tangency while preserving the input
    vector's storage dtype for the optimizer state.
    """

    if point.ndim != 2 or vector.ndim != 2 or point.shape != vector.shape:
        raise ValueError(
            "Stiefel projection needs matching matrices, got "
            f"{tuple(point.shape)} and {tuple(vector.shape)}."
        )
    if point.shape[0] < point.shape[1]:
        raise ValueError(
            "Column-orthonormal Stiefel factors must be tall or square, got "
            f"{tuple(point.shape)}."
        )

    work_dtype = torch.float64 if point.dtype == torch.float64 else torch.float32
    work_point = point.to(dtype=work_dtype)
    work_vector = vector.to(dtype=work_dtype)
    cross = work_point.mT @ work_vector
    symmetric = 0.5 * (cross + cross.mT)
    projected = work_vector - work_point @ symmetric
    return projected.to(dtype=vector.dtype)


def tucker_core_shape_overrides(
    model: torch.nn.Module,
) -> dict[torch.nn.Parameter, tuple[int, ...]]:
    """Expose matrix-stored Tucker cores to Tensorion as logical 4-D tensors.

    ``TuckerLinear.core_matrix`` is stored as ``[r3*r4, r1*r2]`` so Muon sees
    a matrix.  Tensorion should instead see its native four-mode structure.
    Reshaping in storage order gives ``[r3, r4, r1, r2]`` without a copy.
    """

    overrides: dict[torch.nn.Parameter, tuple[int, ...]] = {}
    for module in model.modules():
        core = getattr(module, "core_matrix", None)
        ranks = getattr(module, "ranks", None)
        if not isinstance(core, torch.nn.Parameter) or ranks is None:
            continue
        if len(ranks) != 4:
            continue
        r1, r2, r3, r4 = (int(rank) for rank in ranks)
        logical_shape = (r3, r4, r1, r2)
        if math.prod(logical_shape) != core.numel():
            raise ValueError(
                f"Tucker core shape {logical_shape} does not match "
                f"core_matrix with {core.numel():,} elements."
            )
        overrides[core] = logical_shape
    return overrides


class TensorionOptimizer(torch.optim.Optimizer):
    """Tensorion + Muon + AdamW in one scheduler-compatible optimizer.

    ``tensorion_params`` contains ``(name, parameter, logical_shape)`` tuples.
    ``logical_shape`` may differ from the physical parameter shape, which is
    used for the matrix-stored Tucker cores in this repository.
    """

    def __init__(
        self,
        tensorion_params: Iterable[tuple[str, torch.nn.Parameter, Sequence[int]]],
        adamw_param_groups: Iterable[dict],
        *,
        muon_params: Iterable[tuple[str, torch.nn.Parameter]] = (),
        riemannian_muon_params: Iterable[
            tuple[str, torch.nn.Parameter]
        ] = (),
        tucker_module_specs: Iterable[
            tuple[
                str,
                torch.nn.Parameter,
                tuple[
                    torch.nn.Parameter,
                    torch.nn.Parameter,
                    torch.nn.Parameter,
                    torch.nn.Parameter,
                ],
            ]
        ] = (),
        tucker_lr_scaling_mode: str = "none",
        tucker_lr_scaling_eps: float = 1e-8,
        tucker_lr_scaling_power_iters: int = 1,
        tucker_lr_scaling_use_stiefel_unit_norm: bool = True,
        tucker_lr_scaling_post_ns_project: bool = True,
        tucker_lr_scaling_stiefel_drift_threshold: float = 1e-3,
        tucker_lr_scaling_strict_bound_check: bool = False,
        tucker_lr_scaling_exact_svd_debug: bool = False,
        tucker_lr_scaling_log_interval: int = 100,
        tucker_riemannian_muon_post_ns_project: bool = False,
        parallel_tucker_components: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = False,
        adjust_lr: bool = True,
        ns_steps: int = 5,
        orthogonalization: str = "ns",
        adamw_betas: tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid Tensorion learning rate: {lr}")
        if weight_decay < 0:
            raise ValueError(f"Invalid Tensorion weight decay: {weight_decay}")
        if not 0 <= momentum < 1:
            raise ValueError(f"Invalid Tensorion momentum: {momentum}")
        if orthogonalization not in ("ns", "svd"):
            raise ValueError("orthogonalization must be 'ns' or 'svd'")
        if ns_steps < 1:
            raise ValueError("ns_steps must be >= 1")
        beta1, beta2 = adamw_betas
        if not 0 <= beta1 < 1 or not 0 <= beta2 < 1:
            raise ValueError(f"Invalid AdamW betas: {adamw_betas}")
        if adamw_eps < 0:
            raise ValueError(f"Invalid AdamW epsilon: {adamw_eps}")
        if tucker_lr_scaling_mode not in (
            "none",
            "spectron_bound",
            "first_order",
            "first_order_calibrated",
            "paper_mup",
            "paper_mup_functional",
        ):
            raise ValueError(
                "tucker_lr_scaling_mode must be none, spectron_bound, "
                "first_order, first_order_calibrated, paper_mup, or "
                "paper_mup_functional; "
                f"got {tucker_lr_scaling_mode!r}"
            )
        if tucker_lr_scaling_eps <= 0:
            raise ValueError("tucker_lr_scaling_eps must be positive")
        if tucker_lr_scaling_power_iters < 1:
            raise ValueError("tucker_lr_scaling_power_iters must be >= 1")
        if tucker_lr_scaling_stiefel_drift_threshold <= 0:
            raise ValueError(
                "tucker_lr_scaling_stiefel_drift_threshold must be positive"
            )
        if tucker_lr_scaling_log_interval < 0:
            raise ValueError("tucker_lr_scaling_log_interval must be non-negative")
        if (
            tucker_lr_scaling_strict_bound_check
            and tucker_lr_scaling_mode != "spectron_bound"
        ):
            raise ValueError(
                "strict_bound_check is only valid for spectron_bound mode"
            )

        tensorion_specs = list(tensorion_params)
        muon_specs = list(muon_params)
        riemannian_muon_specs = list(riemannian_muon_params)
        raw_tucker_specs = list(tucker_module_specs)
        parameter_groups = []
        if tensorion_specs:
            parameter_groups.append(
                {
                    "params": [parameter for _, parameter, _ in tensorion_specs],
                    "update_type": "tensorion",
                    "weight_decay": weight_decay,
                }
            )
        if muon_specs:
            parameter_groups.append(
                {
                    "params": [parameter for _, parameter in muon_specs],
                    "update_type": "muon",
                    "weight_decay": weight_decay,
                }
            )
        if riemannian_muon_specs:
            parameter_groups.append(
                {
                    "params": [
                        parameter for _, parameter in riemannian_muon_specs
                    ],
                    "update_type": "riemannian_muon",
                    "weight_decay": weight_decay,
                }
            )
        for source_group in adamw_param_groups:
            params = list(source_group["params"])
            if not params:
                continue
            group = dict(source_group)
            group["params"] = params
            group["update_type"] = "adamw"
            group.setdefault("weight_decay", weight_decay)
            parameter_groups.append(group)
        if not parameter_groups:
            raise ValueError("TensorionOptimizer received no parameters.")

        defaults = dict(lr=lr)
        super().__init__(parameter_groups, defaults)

        self.momentum = float(momentum)
        self.nesterov = bool(nesterov)
        self.adjust_lr = bool(adjust_lr)
        self.ns_steps = int(ns_steps)
        self.orthogonalization = orthogonalization
        self.adamw_betas = (float(beta1), float(beta2))
        self.adamw_eps = float(adamw_eps)
        self.tucker_lr_scaling_mode = tucker_lr_scaling_mode
        self.tucker_lr_scaling_eps = float(tucker_lr_scaling_eps)
        self.tucker_lr_scaling_power_iters = int(tucker_lr_scaling_power_iters)
        self.tucker_lr_scaling_use_stiefel_unit_norm = bool(
            tucker_lr_scaling_use_stiefel_unit_norm
        )
        self.tucker_lr_scaling_post_ns_project = bool(
            tucker_lr_scaling_post_ns_project
        )
        self.tucker_lr_scaling_stiefel_drift_threshold = float(
            tucker_lr_scaling_stiefel_drift_threshold
        )
        self.tucker_lr_scaling_strict_bound_check = bool(
            tucker_lr_scaling_strict_bound_check
        )
        self.tucker_lr_scaling_exact_svd_debug = bool(
            tucker_lr_scaling_exact_svd_debug
        )
        self.tucker_lr_scaling_log_interval = int(
            tucker_lr_scaling_log_interval
        )
        self.tucker_riemannian_muon_post_ns_project = bool(
            tucker_riemannian_muon_post_ns_project
        )
        self.parallel_tucker_components = bool(parallel_tucker_components)
        self._tucker_component_streams = {}
        self._tucker_scaling_step = 0
        self._last_tucker_lr_scaling_metrics: dict[str, float] = {}
        self._stiefel_actual_norm_parameters: set[torch.nn.Parameter] = set()
        self._warned_stiefel_parameters: set[torch.nn.Parameter] = set()
        self._plans: dict[torch.nn.Parameter, UnfoldingPlan] = {}
        self._names: dict[torch.nn.Parameter, str] = {}
        self._muon_plans: dict[torch.nn.Parameter, UnfoldingPlan] = {}
        self._muon_names: dict[torch.nn.Parameter, str] = {}
        self._riemannian_muon_parameters: set[torch.nn.Parameter] = set()
        self._parameter_groups_by_parameter: dict[torch.nn.Parameter, dict] = {}
        for group in self.param_groups:
            for parameter in group["params"]:
                self._parameter_groups_by_parameter[parameter] = group

        for name, parameter, logical_shape in tensorion_specs:
            if parameter in self._plans:
                raise ValueError(f"Tensorion parameter {name!r} was supplied twice.")
            if parameter.numel() != math.prod(tuple(logical_shape)):
                raise ValueError(
                    f"Tensorion shape {tuple(logical_shape)} for {name!r} has the "
                    f"wrong number of elements ({parameter.numel():,})."
                )
            self._plans[parameter] = select_balanced_unfolding(logical_shape)
            self._names[parameter] = name

        tensorion_set = set(self._plans)
        for name, parameter in muon_specs:
            if parameter in tensorion_set or parameter in self._muon_plans:
                raise ValueError(f"Optimizer parameter {name!r} was supplied twice.")
            if parameter.ndim != 2:
                raise ValueError(
                    f"Muon parameter {name!r} must be a matrix, got "
                    f"shape {tuple(parameter.shape)}."
                )
            self._muon_plans[parameter] = select_balanced_unfolding(parameter.shape)
            self._muon_names[parameter] = name

        for name, parameter in riemannian_muon_specs:
            if parameter in tensorion_set or parameter in self._muon_plans:
                raise ValueError(f"Optimizer parameter {name!r} was supplied twice.")
            if parameter.ndim != 2 or parameter.shape[0] < parameter.shape[1]:
                raise ValueError(
                    f"Riemannian Muon parameter {name!r} must be a tall-or-square "
                    f"matrix, got shape {tuple(parameter.shape)}."
                )
            self._muon_plans[parameter] = select_balanced_unfolding(parameter.shape)
            self._muon_names[parameter] = name
            self._riemannian_muon_parameters.add(parameter)

        self._tucker_specs: tuple[TuckerCoupledSpec, ...] = tuple(
            TuckerCoupledSpec(name=name, core=core, factors=tuple(factors))
            for name, core, factors in raw_tucker_specs
        )
        self._coupled_parameters: set[torch.nn.Parameter] = set()
        if self.tucker_lr_scaling_mode != "none":
            for spec in self._tucker_specs:
                if spec.core not in self._plans:
                    raise ValueError(
                        f"Tucker core for {spec.name!r} is not a Tensorion parameter"
                    )
                if len(spec.factors) != 4:
                    raise ValueError(
                        f"Tucker module {spec.name!r} must have four factors"
                    )
                for factor in spec.factors:
                    if factor not in self._riemannian_muon_parameters:
                        raise ValueError(
                            f"Tucker factor for {spec.name!r} is not a "
                            "Riemannian-Muon parameter"
                        )
                linked = (spec.core, *spec.factors)
                overlap = self._coupled_parameters.intersection(linked)
                if overlap:
                    raise ValueError(
                        f"Tucker parameters for {spec.name!r} were supplied twice"
                    )
                self._coupled_parameters.update(linked)

    @property
    def last_tucker_lr_scaling_metrics(self) -> dict[str, float]:
        """Metrics produced at the configured scaler logging cadence."""

        return dict(self._last_tucker_lr_scaling_metrics)

    def load_state_dict(self, state_dict):
        """Restore optimizer state while retaining FP32 power-iteration vectors."""

        result = super().load_state_dict(state_dict)
        # ``Optimizer.load_state_dict`` rebuilds ``param_groups`` from copies of
        # the saved groups, so the cached references would keep the learning
        # rate they held before the resume while the scheduler updates the new
        # dictionaries.
        self._parameter_groups_by_parameter = {
            parameter: group
            for group in self.param_groups
            for parameter in group["params"]
        }
        for state in self.state.values():
            for key, value in tuple(state.items()):
                if key.startswith("spectron_") and isinstance(value, torch.Tensor):
                    state[key] = value.to(dtype=torch.float32)
        return result

    @property
    def unfolding_plans(self) -> dict[str, UnfoldingPlan]:
        return {self._names[param]: plan for param, plan in self._plans.items()}

    @property
    def muon_plans(self) -> dict[str, UnfoldingPlan]:
        return {
            self._muon_names[param]: plan
            for param, plan in self._muon_plans.items()
        }

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._last_tucker_lr_scaling_metrics = {}
        if self.tucker_lr_scaling_mode in (
            "paper_mup",
            "paper_mup_functional",
        ):
            self._tucker_scaling_step += 1
            self._tucker_paper_mup_step()
        elif self.tucker_lr_scaling_mode != "none":
            self._tucker_scaling_step += 1
            self._tucker_scaled_step()
        for group in self.param_groups:
            if group["update_type"] == "tensorion":
                self._tensorion_step(group)
            elif group["update_type"] == "muon":
                self._muon_step(group)
            elif group["update_type"] == "riemannian_muon":
                self._muon_step(group, project_gradient=True)
            else:
                self._adamw_step(group)
        return loss

    def _direction_scale(self, plan: UnfoldingPlan) -> float:
        if not self.adjust_lr:
            return 1.0
        return 0.2 * math.sqrt(max(plan.rows, plan.columns))

    def _scaled_core_direction(
        self,
        parameter: torch.nn.Parameter,
        *,
        apply_shape_scale: bool = True,
    ) -> torch.Tensor:
        grad = parameter.grad
        if grad is None:
            raise RuntimeError("Coupled Tucker core is missing its gradient")
        if grad.is_sparse:
            raise RuntimeError("Tensorion does not support sparse gradients")
        state = self.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(parameter)
        buffer = state["momentum_buffer"]
        buffer.mul_(self.momentum).add_(grad)
        momentum = grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
        plan = self._plans[parameter]
        direction = tensorion_direction(
            momentum,
            plan,
            orthogonalization=self.orthogonalization,
            ns_steps=self.ns_steps,
        )
        if apply_shape_scale:
            direction = direction.mul(self._direction_scale(plan))
        return direction

    def _scaled_factor_direction(
        self,
        parameter: torch.nn.Parameter,
        *,
        apply_shape_scale: bool = True,
    ) -> torch.Tensor:
        grad = parameter.grad
        if grad is None:
            raise RuntimeError("Coupled Tucker factor is missing its gradient")
        if grad.is_sparse:
            raise RuntimeError("Muon does not support sparse gradients")

        # Retain the existing pre-momentum projection so the transported buffer
        # remains tangent. The final Muon-direction projection is configurable
        # for controlled coupled-LR ablations.
        grad = stiefel_tangent_projection(parameter, grad)
        state = self.state[parameter]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = torch.zeros_like(parameter)
        buffer = state["momentum_buffer"]
        buffer.mul_(self.momentum).add_(grad, alpha=1.0 - self.momentum)
        momentum = buffer.mul(self.momentum).add(
            grad,
            alpha=1.0 - self.momentum,
        )
        plan = self._muon_plans[parameter]
        direction = tensorion_direction(
            momentum,
            plan,
            orthogonalization=self.orthogonalization,
            ns_steps=self.ns_steps,
        )
        if self.tucker_lr_scaling_post_ns_project:
            direction = stiefel_tangent_projection(parameter, direction)
        if apply_shape_scale:
            direction = direction.mul(self._direction_scale(plan))
        return direction

    def _scaled_tucker_directions(
        self,
        spec: TuckerCoupledSpec,
        *,
        apply_shape_scale: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        if not self.parallel_tucker_components or not spec.core.is_cuda:
            return (
                self._scaled_core_direction(
                    spec.core,
                    apply_shape_scale=apply_shape_scale,
                ),
                *(
                    self._scaled_factor_direction(
                        factor,
                        apply_shape_scale=apply_shape_scale,
                    )
                    for factor in spec.factors
                ),
            )

        device = spec.core.device
        streams = self._tucker_component_streams.get(device)
        if streams is None:
            streams = tuple(torch.cuda.Stream(device=device) for _ in range(5))
            self._tucker_component_streams[device] = streams
        current_stream = torch.cuda.current_stream(device)
        for stream in streams:
            stream.wait_stream(current_stream)

        directions = []
        for index, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                if index == 0:
                    direction = self._scaled_core_direction(
                        spec.core,
                        apply_shape_scale=apply_shape_scale,
                    )
                else:
                    direction = self._scaled_factor_direction(
                        spec.factors[index - 1],
                        apply_shape_scale=apply_shape_scale,
                    )
                directions.append(direction)

        for stream, direction in zip(streams, directions):
            current_stream.wait_stream(stream)
            direction.record_stream(current_stream)
        return tuple(directions)

    @staticmethod
    def _stiefel_orthogonality_error(parameter: torch.Tensor) -> torch.Tensor:
        work = parameter.detach().to(dtype=torch.float32)
        gram = work.mT @ work
        identity = torch.eye(gram.shape[0], device=gram.device, dtype=gram.dtype)
        return torch.linalg.vector_norm(gram - identity) / math.sqrt(gram.shape[0])

    def _factor_sigma(
        self,
        parameter: torch.nn.Parameter,
        *,
        log_due: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        orthogonality_error = None
        if log_due:
            orthogonality_error = self._stiefel_orthogonality_error(parameter)
            if (
                orthogonality_error
                > self.tucker_lr_scaling_stiefel_drift_threshold
            ):
                self._stiefel_actual_norm_parameters.add(parameter)
                if parameter not in self._warned_stiefel_parameters:
                    warnings.warn(
                        "Tucker Stiefel drift exceeded the configured threshold; "
                        "falling back to an estimated factor spectral norm.",
                        RuntimeWarning,
                    )
                    self._warned_stiefel_parameters.add(parameter)

        use_unit = (
            self.tucker_lr_scaling_use_stiefel_unit_norm
            and parameter not in self._stiefel_actual_norm_parameters
        )
        if use_unit:
            sigma = torch.ones((), device=parameter.device, dtype=torch.float32)
        else:
            sigma = warm_started_spectral_norm(
                parameter,
                self.state[parameter],
                prefix="spectron_weight",
                power_iters=self.tucker_lr_scaling_power_iters,
                eps=self.tucker_lr_scaling_eps,
                exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
            )
        return sigma, orthogonality_error

    @staticmethod
    def _materialize_small_tucker(
        core: torch.Tensor,
        factors: Sequence[torch.Tensor],
    ) -> torch.Tensor:
        if len(factors) != 4:
            raise ValueError("Strict Tucker check expects four factors")
        U1, U2, U3, U4 = (factor.to(dtype=torch.float32) for factor in factors)
        G = core.to(dtype=torch.float32)
        rows = U3.shape[0] * U4.shape[0]
        columns = U1.shape[0] * U2.shape[0]
        if rows * columns > 1_000_000:
            raise RuntimeError(
                "strict_bound_check only materializes debug operators with at "
                "most 1,000,000 elements"
            )
        return torch.kron(U3, U4) @ G @ torch.kron(U1, U2).mT

    def _tucker_scaled_step(self) -> None:
        if not self._tucker_specs:
            raise RuntimeError(
                "Tucker LR scaling is enabled but no coupled Tucker modules exist"
            )
        log_due = bool(
            self.tucker_lr_scaling_log_interval
            and self._tucker_scaling_step % self.tucker_lr_scaling_log_interval == 0
        )
        selected_indices = {0, len(self._tucker_specs) // 2, len(self._tucker_specs) - 1}
        metrics: dict[str, float] = {}

        for spec_index, spec in enumerate(self._tucker_specs):
            linked = (spec.core, *spec.factors)
            gradients_present = [parameter.grad is not None for parameter in linked]
            if not any(gradients_present):
                continue
            if not all(gradients_present):
                raise RuntimeError(
                    f"Tucker module {spec.name!r} has only a partial gradient set"
                )

            groups = [self._parameter_groups_by_parameter[parameter] for parameter in linked]
            base_lr = float(groups[0]["lr"])
            if any(not math.isclose(float(group["lr"]), base_lr) for group in groups[1:]):
                raise RuntimeError(
                    f"Coupled Tucker module {spec.name!r} has inconsistent group LRs"
                )

            strict_before = None
            if self.tucker_lr_scaling_strict_bound_check:
                strict_before = self._materialize_small_tucker(
                    spec.core.detach().clone(),
                    [factor.detach().clone() for factor in spec.factors],
                )

            core_direction, *factor_directions = self._scaled_tucker_directions(spec)
            # In scaled modes the four Stiefel factors receive no normal-space
            # decay.  Include layer weight decay once in the core direction so
            # it is covered by the same reconstructed-weight spectral budget.
            core_weight_decay = float(groups[0].get("weight_decay", 0.0))
            if core_weight_decay:
                core_direction = core_direction.add(
                    spec.core,
                    alpha=core_weight_decay,
                )
            core_sigma = warm_started_spectral_norm(
                spec.core,
                self.state[spec.core],
                prefix="spectron_weight",
                power_iters=self.tucker_lr_scaling_power_iters,
                eps=self.tucker_lr_scaling_eps,
                exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
            )
            core_rho = warm_started_spectral_norm(
                core_direction,
                self.state[spec.core],
                prefix="spectron_direction",
                power_iters=self.tucker_lr_scaling_power_iters,
                eps=self.tucker_lr_scaling_eps,
                exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
            )
            factor_sigmas = []
            factor_rhos = []
            orthogonality_errors = []
            for factor, direction in zip(spec.factors, factor_directions):
                sigma, orthogonality_error = self._factor_sigma(
                    factor,
                    log_due=log_due,
                )
                factor_sigmas.append(sigma)
                if orthogonality_error is not None:
                    orthogonality_errors.append(orthogonality_error)
                factor_rhos.append(
                    warm_started_spectral_norm(
                        direction,
                        self.state[factor],
                        prefix="spectron_direction",
                        power_iters=self.tucker_lr_scaling_power_iters,
                        eps=self.tucker_lr_scaling_eps,
                        exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
                    )
                )

            denominator_mode = (
                "first_order"
                if self.tucker_lr_scaling_mode == "first_order_calibrated"
                else self.tucker_lr_scaling_mode
            )
            denominator = tucker_spectral_denominator(
                [core_sigma, *factor_sigmas],
                [core_rho, *factor_rhos],
                mode=denominator_mode,
            ).clamp_min(0.0)
            eta = torch.as_tensor(
                base_lr,
                device=spec.core.device,
                dtype=torch.float32,
            )
            reference_denominator = None
            if self.tucker_lr_scaling_mode == "first_order_calibrated":
                state = self.state[spec.core]
                reference_denominator = state.get(
                    "tucker_first_order_reference_denominator"
                )
                if reference_denominator is None:
                    # Calibrate the first step to the exact legacy coefficient.
                    # Therefore the candidate component updates, and hence the
                    # actual reconstructed-weight change, initially match the
                    # unscaled default run. Later steps adapt only to relative
                    # changes in the first-order functional sensitivity.
                    reference_denominator = denominator.detach().clone()
                    state["tucker_first_order_reference_denominator"] = (
                        reference_denominator
                    )
                else:
                    reference_denominator = reference_denominator.to(
                        device=denominator.device,
                        dtype=denominator.dtype,
                    )
                eta = eta * reference_denominator
            alpha = torch.minimum(
                torch.ones_like(eta),
                eta / (denominator + self.tucker_lr_scaling_eps),
            )

            spec.core.add_(
                core_direction.mul(alpha.to(dtype=core_direction.dtype)),
                alpha=-1.0,
            )
            for factor, direction in zip(spec.factors, factor_directions):
                factor.add_(
                    direction.mul(alpha.to(dtype=direction.dtype)),
                    alpha=-1.0,
                )

            if strict_before is not None:
                strict_after = self._materialize_small_tucker(
                    spec.core,
                    spec.factors,
                )
                actual_delta = torch.linalg.matrix_norm(
                    strict_after - strict_before,
                    ord=2,
                )
                if actual_delta > base_lr + 1e-5:
                    raise RuntimeError(
                        f"Strict Tucker spectral bound failed for {spec.name!r}: "
                        f"{float(actual_delta):.6g} > {base_lr:.6g}"
                    )

            if log_due and spec_index in selected_indices:
                key = spec.name.replace(".", "/") or "root"
                prefix = f"tucker_lr_scaling/{key}"
                metrics[f"{prefix}/base_lr"] = base_lr
                metrics[f"{prefix}/alpha"] = float(alpha.cpu())
                metrics[f"{prefix}/denominator"] = float(denominator.cpu())
                metrics[f"{prefix}/core_sigma"] = float(core_sigma.cpu())
                metrics[f"{prefix}/core_rho"] = float(core_rho.cpu())
                metrics[f"{prefix}/alpha_over_base_lr"] = (
                    float(alpha.cpu()) / base_lr if base_lr else 0.0
                )
                metrics[f"{prefix}/analytic_delta_bound"] = float(
                    (alpha * denominator).cpu()
                )
                if reference_denominator is not None:
                    metrics[f"{prefix}/reference_denominator"] = float(
                        reference_denominator.cpu()
                    )
                    metrics[f"{prefix}/calibrated_budget"] = float(eta.cpu())
                for factor_index, rho in enumerate(factor_rhos, start=1):
                    metrics[f"{prefix}/rho_U{factor_index}"] = float(rho.cpu())
                if orthogonality_errors:
                    metrics[f"{prefix}/max_stiefel_orthogonality_error"] = float(
                        torch.stack(orthogonality_errors).max().cpu()
                    )

        self._last_tucker_lr_scaling_metrics = metrics

    def _tucker_paper_mup_step(self) -> None:
        """Apply Qiu et al.'s static component-wise structure-aware LR rule.

        ``paper_mup`` preserves the historical implementation. The corrected
        ``paper_mup_functional`` variant applies kappa to the raw normalized
        Tensorion/Muon directions (without a second ``0.2 * sqrt(d)`` factor)
        and rescales all kappas by one layer-wise scalar so the first-order
        reconstructed-weight update proxy matches the legacy default run.
        """

        if not self._tucker_specs:
            raise RuntimeError(
                "Paper-muP Tucker LR scaling is enabled but no coupled "
                "Tucker modules exist"
            )
        log_due = bool(
            self.tucker_lr_scaling_log_interval
            and self._tucker_scaling_step % self.tucker_lr_scaling_log_interval == 0
        )
        selected_indices = {0, len(self._tucker_specs) // 2, len(self._tucker_specs) - 1}
        metrics: dict[str, float] = {}

        for spec_index, spec in enumerate(self._tucker_specs):
            linked = (spec.core, *spec.factors)
            gradients_present = [parameter.grad is not None for parameter in linked]
            if not any(gradients_present):
                continue
            if not all(gradients_present):
                raise RuntimeError(
                    f"Tucker module {spec.name!r} has only a partial gradient set"
                )

            groups = [self._parameter_groups_by_parameter[parameter] for parameter in linked]
            base_lr = float(groups[0]["lr"])
            if any(not math.isclose(float(group["lr"]), base_lr) for group in groups[1:]):
                raise RuntimeError(
                    f"Coupled Tucker module {spec.name!r} has inconsistent group LRs"
                )

            functional_mode = self.tucker_lr_scaling_mode == "paper_mup_functional"
            direction_scales = (
                self._direction_scale(self._plans[spec.core]),
                *(
                    self._direction_scale(self._muon_plans[factor])
                    for factor in spec.factors
                ),
            )
            directions = self._scaled_tucker_directions(
                spec,
                apply_shape_scale=not functional_mode,
            )
            raw_multipliers = tucker_paper_mup_lr_multipliers(
                spec.core,
                spec.factors,
            )
            functional_normalizer = 1.0
            legacy_proxy = None
            paper_proxy = None
            if functional_mode:
                core_sigma = warm_started_spectral_norm(
                    spec.core,
                    self.state[spec.core],
                    prefix="paper_functional_weight",
                    power_iters=self.tucker_lr_scaling_power_iters,
                    eps=self.tucker_lr_scaling_eps,
                    exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
                )
                factor_sigmas = [
                    self._factor_sigma(factor, log_due=log_due)[0]
                    for factor in spec.factors
                ]
                sigmas = [core_sigma, *factor_sigmas]
                direction_rhos = [
                    warm_started_spectral_norm(
                        direction,
                        self.state[parameter],
                        prefix="paper_functional_direction",
                        power_iters=self.tucker_lr_scaling_power_iters,
                        eps=self.tucker_lr_scaling_eps,
                        exact_svd_debug=self.tucker_lr_scaling_exact_svd_debug,
                    )
                    for parameter, direction in zip(linked, directions)
                ]
                sensitivities = []
                for component_index in range(len(sigmas)):
                    other_sigmas = (
                        sigmas[:component_index] + sigmas[component_index + 1 :]
                    )
                    sensitivities.append(
                        direction_rhos[component_index]
                        * torch.stack(other_sigmas).prod()
                    )
                legacy_proxy = sum(
                    sensitivity * scale
                    for sensitivity, scale in zip(
                        sensitivities,
                        direction_scales,
                    )
                )
                paper_proxy = sum(
                    sensitivity * multiplier
                    for sensitivity, multiplier in zip(
                        sensitivities,
                        raw_multipliers,
                    )
                )
                functional_normalizer = float(
                    (
                        legacy_proxy
                        / (paper_proxy + self.tucker_lr_scaling_eps)
                    ).cpu()
                )
            multipliers = tuple(
                multiplier * functional_normalizer
                for multiplier in raw_multipliers
            )
            effective_lrs = tuple(base_lr * multiplier for multiplier in multipliers)

            for parameter, direction, group, effective_lr in zip(
                linked,
                directions,
                groups,
                effective_lrs,
            ):
                weight_decay = float(group.get("weight_decay", 0.0))
                if weight_decay:
                    decay_lr = base_lr if functional_mode else effective_lr
                    parameter.mul_(1.0 - decay_lr * weight_decay)
                parameter.add_(direction, alpha=-effective_lr)

            if log_due and spec_index in selected_indices:
                key = spec.name.replace(".", "/") or "root"
                prefix = f"tucker_lr_scaling/{key}"
                labels = ("core", "U1", "U2", "U3", "U4")
                metrics[f"{prefix}/base_lr"] = base_lr
                metrics[f"{prefix}/paper_component_count"] = 5.0
                metrics[f"{prefix}/functional_normalizer"] = float(
                    functional_normalizer
                )
                if legacy_proxy is not None and paper_proxy is not None:
                    metrics[f"{prefix}/legacy_functional_proxy"] = float(
                        legacy_proxy.cpu()
                    )
                    metrics[f"{prefix}/raw_paper_functional_proxy"] = float(
                        paper_proxy.cpu()
                    )
                for label, raw_multiplier, multiplier, effective_lr, direction_scale in zip(
                    labels,
                    raw_multipliers,
                    multipliers,
                    effective_lrs,
                    direction_scales,
                ):
                    metrics[f"{prefix}/paper_kappa_{label}"] = float(raw_multiplier)
                    metrics[f"{prefix}/normalized_kappa_{label}"] = float(multiplier)
                    metrics[f"{prefix}/effective_lr_{label}"] = float(effective_lr)
                    metrics[f"{prefix}/applied_step_coefficient_{label}"] = float(
                        effective_lr
                        if functional_mode
                        else effective_lr * direction_scale
                    )
                    metrics[f"{prefix}/legacy_step_coefficient_{label}"] = float(
                        base_lr * direction_scale
                    )
                orthogonality_errors = [
                    self._stiefel_orthogonality_error(factor)
                    for factor in spec.factors
                ]
                metrics[f"{prefix}/max_stiefel_orthogonality_error"] = float(
                    torch.stack(orthogonality_errors).max().cpu()
                )

        self._last_tucker_lr_scaling_metrics = metrics

    def _tensorion_step(self, group: dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        for parameter in group["params"]:
            if parameter in self._coupled_parameters:
                continue
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Tensorion does not support sparse gradients.")

            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer = state["momentum_buffer"]
            buffer.mul_(self.momentum).add_(grad)
            momentum = (
                grad.add(buffer, alpha=self.momentum) if self.nesterov else buffer
            )

            plan = self._plans[parameter]
            update = tensorion_direction(
                momentum,
                plan,
                orthogonalization=self.orthogonalization,
                ns_steps=self.ns_steps,
            )
            step_lr = lr
            if self.adjust_lr:
                step_lr *= 0.2 * math.sqrt(max(plan.rows, plan.columns))

            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            parameter.add_(update, alpha=-step_lr)

    def _muon_step(self, group: dict, *, project_gradient: bool = False) -> None:
        """Apply the standard Nesterov Muon update to eligible matrices."""

        lr = group["lr"]
        weight_decay = group["weight_decay"]
        for parameter in group["params"]:
            if parameter in self._coupled_parameters:
                continue
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError("Muon does not support sparse gradients.")
            if project_gradient:
                grad = stiefel_tangent_projection(parameter, grad)

            state = self.state[parameter]
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
            buffer = state["momentum_buffer"]
            buffer.mul_(self.momentum).add_(grad, alpha=1.0 - self.momentum)
            momentum = buffer.mul(self.momentum).add(
                grad,
                alpha=1.0 - self.momentum,
            )

            plan = self._muon_plans[parameter]
            update = tensorion_direction(
                momentum,
                plan,
                orthogonalization=self.orthogonalization,
                ns_steps=self.ns_steps,
            )
            if project_gradient and self.tucker_riemannian_muon_post_ns_project:
                update = stiefel_tangent_projection(parameter, update)
            step_lr = lr
            if self.adjust_lr:
                step_lr *= 0.2 * math.sqrt(max(plan.rows, plan.columns))

            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            parameter.add_(update, alpha=-step_lr)

    def _adamw_step(self, group: dict) -> None:
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = self.adamw_betas

        for parameter in group["params"]:
            grad = parameter.grad
            if grad is None:
                continue
            if grad.is_sparse:
                raise RuntimeError(
                    "Tensorion's AdamW fallback does not support sparse gradients."
                )

            state = self.state[parameter]
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            if weight_decay:
                parameter.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2))
            denominator.add_(self.adamw_eps)
            parameter.addcdiv_(
                exp_avg,
                denominator,
                value=-lr / bias_correction1,
            )


__all__ = [
    "TensorionOptimizer",
    "TuckerCoupledSpec",
    "UnfoldingPlan",
    "fold_tensor",
    "select_balanced_unfolding",
    "stiefel_tangent_projection",
    "tensorion_direction",
    "tucker_core_shape_overrides",
    "unfold_tensor",
]
