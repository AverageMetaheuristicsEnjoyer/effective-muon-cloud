"""Function-preserving progressive rank growth for Tucker language models.

The controller in this module grows the ranks of existing ``TuckerLinear``
parameters while preserving their positions in optimizer parameter groups and
checkpoint ordering.  Progressive mode is intentionally restricted to a
single, uncompiled process because DDP reducers and compiled graphs cache
parameter shapes.

For an expansion ``r -> r'`` the factors become ``[U, Q]``, where ``Q`` is an
orthonormal complement, and the old core is copied into the leading block of a
zero-filled larger core.  Consequently the represented dense matrix is
unchanged at the instant of growth.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

import torch


@contextmanager
def _exact_fp32_matmul():
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


RankTuple = tuple[int, int, int, int]
RankPlan = dict[str, RankTuple]


@dataclass(frozen=True)
class ProgressiveStage:
    step: int
    target_parameters: int
    actual_parameters: int
    ranks: RankPlan


def parse_progressive_stages(values: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Parse ``STEP:PARAMETERS`` CLI values and validate their ordering."""

    parsed: list[tuple[int, int]] = []
    for value in values:
        try:
            raw_step, raw_target = value.split(":", maxsplit=1)
            step, target = int(raw_step), int(raw_target)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Each --tucker-progressive-stages value must be STEP:PARAMETERS"
            ) from error
        if step < 0 or target <= 0:
            raise ValueError("Progressive steps must be non-negative and targets positive")
        parsed.append((step, target))

    if not parsed or parsed[0][0] != 0:
        raise ValueError("The first progressive Tucker stage must start at step 0")
    if any(right[0] <= left[0] for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Progressive Tucker steps must be strictly increasing")
    if any(right[1] <= left[1] for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Progressive Tucker parameter targets must strictly increase")
    return tuple(parsed)


def _tucker_modules(model: torch.nn.Module) -> dict[str, torch.nn.Module]:
    from models.tucker_linear import TuckerLinear

    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, TuckerLinear)
    }
    if not modules:
        raise ValueError("Progressive Tucker mode found no TuckerLinear modules")
    return modules


def _model_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _parameter_count_for_plan(
    model: torch.nn.Module,
    modules: Mapping[str, torch.nn.Module],
    ranks: Mapping[str, RankTuple],
) -> int:
    current_tucker = sum(module.tucker_parameter_count for module in modules.values())
    planned_tucker = 0
    for name, module in modules.items():
        module_ranks = ranks[name]
        planned_tucker += sum(
            int(mode) * int(rank)
            for mode, rank in zip(module.modes, module_ranks)
        ) + math.prod(module_ranks)
    return _model_parameter_count(model) - current_tucker + planned_tucker


def _interpolated_plan(
    start: Mapping[str, RankTuple],
    full: Mapping[str, RankTuple],
    fraction_index: int,
    fraction_denominator: int,
) -> RankPlan:
    plan: RankPlan = {}
    for name, start_ranks in start.items():
        full_ranks = full[name]
        values = []
        for old, maximum in zip(start_ranks, full_ranks):
            # Integer half-up interpolation is deterministic on every Python version.
            numerator = (maximum - old) * fraction_index
            increment = (2 * numerator + fraction_denominator) // (
                2 * fraction_denominator
            )
            values.append(old + increment)
        plan[name] = tuple(values)  # type: ignore[assignment]
    return plan


def build_progressive_rank_stages(
    model: torch.nn.Module,
    stage_specs: Sequence[tuple[int, int]],
    *,
    search_resolution: int = 4096,
) -> tuple[ProgressiveStage, ...]:
    """Find monotone proportional rank plans nearest the requested budgets."""

    if search_resolution < 1:
        raise ValueError("search_resolution must be positive")
    modules = _tucker_modules(model)
    for name, module in modules.items():
        if module.equal_params or module.residual_parameter_count:
            raise ValueError(
                f"Progressive rank growth requires pure Tucker mode; {name!r} "
                "has an equal-parameter residual"
            )

    start: RankPlan = {
        name: tuple(int(rank) for rank in module.ranks)
        for name, module in modules.items()
    }
    full: RankPlan = {
        name: tuple(int(mode) for mode in module.modes)
        for name, module in modules.items()
    }
    current_parameters = _model_parameter_count(model)
    full_parameters = _parameter_count_for_plan(model, modules, full)
    if stage_specs[-1][1] > full_parameters:
        raise ValueError(
            f"Final progressive target {stage_specs[-1][1]:,} exceeds the "
            f"full-rank model size {full_parameters:,}"
        )

    candidates: list[tuple[int, RankPlan]] = []
    last_plan: RankPlan | None = None
    for index in range(search_resolution + 1):
        plan = _interpolated_plan(start, full, index, search_resolution)
        if plan == last_plan:
            continue
        candidates.append((_parameter_count_for_plan(model, modules, plan), plan))
        last_plan = plan

    stages = [
        ProgressiveStage(
            step=stage_specs[0][0],
            target_parameters=stage_specs[0][1],
            actual_parameters=current_parameters,
            ranks=start,
        )
    ]
    previous_actual = current_parameters
    for step, target in stage_specs[1:]:
        eligible = [candidate for candidate in candidates if candidate[0] >= previous_actual]
        actual, plan = min(
            eligible,
            key=lambda item: (
                abs(item[0] - target),
                item[0] > target,
                item[0],
            ),
        )
        stages.append(
            ProgressiveStage(
                step=step,
                target_parameters=target,
                actual_parameters=actual,
                ranks={name: tuple(values) for name, values in plan.items()},
            )
        )
        previous_actual = actual

    for left, right in zip(stages, stages[1:]):
        for name in start:
            if any(a > b for a, b in zip(left.ranks[name], right.ranks[name])):
                raise AssertionError("Rank-plan search produced a non-monotone stage")
    return tuple(stages)


def _orthogonal_complement(
    factor: torch.Tensor,
    target_rank: int,
    *,
    seed: int,
) -> torch.Tensor:
    rows, old_rank = factor.shape
    if not old_rank <= target_rank <= rows:
        raise ValueError(
            f"Cannot expand factor {tuple(factor.shape)} to rank {target_rank}"
        )
    if target_rank == old_rank:
        return factor.detach().clone()

    work_dtype = torch.float64 if factor.dtype == torch.float64 else torch.float32
    work = factor.detach().to(dtype=work_dtype)
    generator = torch.Generator(device=factor.device)
    generator.manual_seed(int(seed))
    needed = target_rank - old_rank
    candidate = torch.randn(
        rows,
        needed,
        generator=generator,
        device=factor.device,
        dtype=work_dtype,
    )
    # Two projections are cheap here and noticeably improve BF16 orthogonality.
    candidate = candidate - work @ (work.mT @ candidate)
    candidate = candidate - work @ (work.mT @ candidate)
    complement, triangular = torch.linalg.qr(candidate, mode="reduced")
    signs = torch.sign(torch.diagonal(triangular))
    signs[signs == 0] = 1.0
    complement = complement * signs
    # Keep the existing columns bit-for-bit; only the newly sampled complement
    # passes through FP32 QR.
    return torch.cat(
        (factor.detach().clone(), complement.to(dtype=factor.dtype)),
        dim=1,
    )


def _expanded_core(
    core_matrix: torch.Tensor,
    old_ranks: RankTuple,
    new_ranks: RankTuple,
) -> torch.Tensor:
    old_r1, old_r2, old_r3, old_r4 = old_ranks
    new_r1, new_r2, new_r3, new_r4 = new_ranks
    old_core = core_matrix.detach().reshape(old_r3, old_r4, old_r1, old_r2)
    new_core = core_matrix.new_zeros((new_r3, new_r4, new_r1, new_r2))
    new_core[:old_r3, :old_r4, :old_r1, :old_r2].copy_(old_core)
    return new_core.reshape(new_r3 * new_r4, new_r1 * new_r2)


def _clear_shape_dependent_scaler_state(state: dict) -> None:
    for key in tuple(state):
        if (
            key.startswith("spectron_")
            or key.startswith("paper_functional_")
            or key == "tucker_first_order_reference_denominator"
        ):
            del state[key]


def _resize_factor_optimizer_state(
    optimizer,
    parameter: torch.nn.Parameter,
    old_shape: tuple[int, int],
    new_shape: tuple[int, int],
) -> None:
    if optimizer is None:
        return
    state = optimizer.state.get(parameter)
    if not state:
        return
    _clear_shape_dependent_scaler_state(state)
    for key, value in tuple(state.items()):
        if isinstance(value, torch.Tensor) and tuple(value.shape) == old_shape:
            expanded = value.new_zeros(new_shape)
            expanded[:, : old_shape[1]].copy_(value)
            state[key] = expanded


def _resize_core_optimizer_state(
    optimizer,
    parameter: torch.nn.Parameter,
    old_ranks: RankTuple,
    new_ranks: RankTuple,
) -> None:
    if optimizer is None:
        return
    state = optimizer.state.get(parameter)
    if not state:
        return
    _clear_shape_dependent_scaler_state(state)
    old_shape = (old_ranks[2] * old_ranks[3], old_ranks[0] * old_ranks[1])
    for key, value in tuple(state.items()):
        if isinstance(value, torch.Tensor) and tuple(value.shape) == old_shape:
            state[key] = _expanded_core(value, old_ranks, new_ranks)


def _replace_parameter_(
    module: torch.nn.Module,
    attribute: str,
    value: torch.Tensor,
    optimizer,
) -> torch.nn.Parameter:
    old = getattr(module, attribute)
    new = torch.nn.Parameter(value, requires_grad=old.requires_grad)
    setattr(module, attribute, new)
    if optimizer is None:
        return new

    replaced = False
    for group in optimizer.param_groups:
        for index, parameter in enumerate(group["params"]):
            if parameter is old:
                group["params"][index] = new
                replaced = True
    if not replaced:
        raise RuntimeError("Expanded Tucker parameter is absent from optimizer groups")

    marker = object()
    state = optimizer.state.pop(old, marker)
    if state is not marker:
        optimizer.state[new] = state

    for name in (
        "_plans",
        "_names",
        "_muon_plans",
        "_muon_names",
        "_parameter_groups_by_parameter",
    ):
        mapping = getattr(optimizer, name, None)
        if mapping is not None and old in mapping:
            mapping[new] = mapping.pop(old)
    for name in (
        "_stiefel_actual_norm_parameters",
        "_warned_stiefel_parameters",
        "_riemannian_muon_parameters",
        "_coupled_parameters",
    ):
        collection = getattr(optimizer, name, None)
        if collection is not None and old in collection:
            collection.remove(old)
            collection.add(new)
    if hasattr(optimizer, "_tucker_specs"):
        optimizer._tucker_specs = tuple(
            replace(
                spec,
                core=new if spec.core is old else spec.core,
                factors=tuple(
                    new if factor is old else factor for factor in spec.factors
                ),
            )
            for spec in optimizer._tucker_specs
        )
    return new


def _refresh_optimizer_structure(optimizer, module: torch.nn.Module) -> None:
    """Refresh Tensorion/Muon unfolding metadata after an in-place resize."""

    if optimizer is None:
        return
    try:
        from optim.tensorion import select_balanced_unfolding
    except ImportError:
        return
    r1, r2, r3, r4 = module.ranks
    if hasattr(optimizer, "_plans") and module.core_matrix in optimizer._plans:
        optimizer._plans[module.core_matrix] = select_balanced_unfolding(
            (r3, r4, r1, r2)
        )
    if hasattr(optimizer, "_muon_plans"):
        for factor in (module.U1, module.U2, module.U3, module.U4):
            if factor in optimizer._muon_plans:
                optimizer._muon_plans[factor] = select_balanced_unfolding(
                    factor.shape
                )
    for attribute in (
        "_stiefel_actual_norm_parameters",
        "_warned_stiefel_parameters",
    ):
        collection = getattr(optimizer, attribute, None)
        if collection is not None:
            for factor in (module.U1, module.U2, module.U3, module.U4):
                collection.discard(factor)


def _update_module_metadata(module: torch.nn.Module, ranks: RankTuple) -> None:
    module.ranks = tuple(int(value) for value in ranks)
    module.rank_policy = ",".join(str(value) for value in ranks)
    module.tucker_parameter_count = sum(
        mode * rank for mode, rank in zip(module.modes, ranks)
    ) + math.prod(ranks)


def _update_model_metadata(model: torch.nn.Module) -> None:
    modules = _tucker_modules(model)
    stats = getattr(model, "_tucker_replacement_stats", None)
    if stats is not None:
        plan_counts: dict[tuple[tuple[int, int], RankTuple, int], int] = {}
        for module in modules.values():
            key = (
                (module.in_features, module.out_features),
                tuple(module.ranks),
                module.residual_parameter_count,
            )
            plan_counts[key] = plan_counts.get(key, 0) + 1
        actual = _model_parameter_count(model)
        model._tucker_replacement_stats = replace(
            stats,
            parameters_after=actual,
            tucker_parameters=sum(
                module.tucker_parameter_count for module in modules.values()
            ),
            target_parameter_count=actual,
            parameter_difference_from_target=0,
            plans=tuple(
                (shape, ranks, residual, count)
                for (shape, ranks, residual), count in sorted(plan_counts.items())
            ),
        )
    for cache_name in ("_num_fwd_flops", "_num_bck_flops"):
        if hasattr(model, cache_name):
            delattr(model, cache_name)


@torch.no_grad()
def expand_tucker_model_to_plan_(
    model: torch.nn.Module,
    optimizer,
    ranks: Mapping[str, Sequence[int]],
    *,
    seed: int,
    verify_function: bool = True,
    verify_rtol: float = 5e-5,
) -> dict[str, float | int]:
    """Expand every requested Tucker module and migrate optimizer references."""

    modules = _tucker_modules(model)
    if set(ranks) != set(modules):
        raise ValueError("Progressive rank plan must name every Tucker module exactly")
    max_absolute_error = 0.0
    max_relative_error = 0.0
    expanded_modules = 0

    for module_index, (name, module) in enumerate(modules.items()):
        old_ranks: RankTuple = tuple(int(value) for value in module.ranks)
        new_ranks: RankTuple = tuple(int(value) for value in ranks[name])  # type: ignore[assignment]
        if len(new_ranks) != 4 or any(old > new for old, new in zip(old_ranks, new_ranks)):
            raise ValueError(
                f"Progressive ranks for {name!r} must grow monotonically: "
                f"{old_ranks} -> {new_ranks}"
            )
        if any(new > mode for new, mode in zip(new_ranks, module.modes)):
            raise ValueError(f"Progressive ranks for {name!r} exceed {module.modes}")
        if old_ranks == new_ranks:
            continue
        if any(
            parameter.grad is not None
            for parameter in (
                module.core_matrix,
                module.U1,
                module.U2,
                module.U3,
                module.U4,
            )
        ):
            raise RuntimeError("Tucker ranks may only grow between optimizer steps")

        with _exact_fp32_matmul():
            before = (
                module.materialize_weight(dtype=torch.float32)
                if verify_function
                else None
            )
        factor_names = ("U1", "U2", "U3", "U4")
        for mode_index, (factor_name, target_rank) in enumerate(
            zip(factor_names, new_ranks)
        ):
            factor = getattr(module, factor_name)
            old_shape = tuple(factor.shape)
            expanded_factor = _orthogonal_complement(
                factor,
                target_rank,
                seed=seed + 1009 * module_index + 97 * mode_index,
            )
            _resize_factor_optimizer_state(
                optimizer,
                factor,
                old_shape,
                tuple(expanded_factor.shape),
            )
            _replace_parameter_(
                module,
                factor_name,
                expanded_factor,
                optimizer,
            )

        old_core = module.core_matrix
        expanded_core = _expanded_core(
            module.core_matrix,
            old_ranks,
            new_ranks,
        )
        _resize_core_optimizer_state(
            optimizer,
            old_core,
            old_ranks,
            new_ranks,
        )
        _replace_parameter_(
            module,
            "core_matrix",
            expanded_core,
            optimizer,
        )
        _update_module_metadata(module, new_ranks)
        _refresh_optimizer_structure(optimizer, module)
        expanded_modules += 1

        if before is not None:
            with _exact_fp32_matmul():
                after = module.materialize_weight(dtype=torch.float32)
            delta = after - before
            absolute = float(delta.abs().max().cpu())
            relative = float(
                (
                    torch.linalg.vector_norm(delta)
                    / torch.linalg.vector_norm(before).clamp_min(1e-12)
                ).cpu()
            )
            max_absolute_error = max(max_absolute_error, absolute)
            max_relative_error = max(max_relative_error, relative)
            if relative > verify_rtol:
                raise RuntimeError(
                    f"Function-preserving Tucker growth failed for {name!r}: "
                    f"relative error {relative:.3e} > {verify_rtol:.3e}"
                )

    _update_model_metadata(model)
    return {
        "expanded_modules": expanded_modules,
        "parameters": _model_parameter_count(model),
        "max_absolute_function_error": max_absolute_error,
        "max_relative_function_error": max_relative_error,
    }


def restore_progressive_tucker_shapes_(
    model: torch.nn.Module,
    optimizer,
    state: Mapping,
) -> None:
    """Resize a freshly built initial-stage model before loading a checkpoint."""

    raw_ranks = state.get("module_ranks")
    if not isinstance(raw_ranks, Mapping):
        raise ValueError("Progressive Tucker checkpoint has no module rank plan")
    ranks = {
        str(name): tuple(int(value) for value in values)
        for name, values in raw_ranks.items()
    }
    expand_tucker_model_to_plan_(
        model,
        optimizer,
        ranks,
        seed=int(state.get("seed", 0)),
        verify_function=False,
    )
    model._progressive_tucker_state = dict(state)


class ProgressiveTuckerController:
    """Advance a pure Tucker model through a fixed parameter-budget schedule."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer,
        stage_values: Sequence[str],
        *,
        warmup_steps: int = 400,
        seed: int = 1701,
        verify_rtol: float = 5e-5,
    ) -> None:
        if warmup_steps < 0:
            raise ValueError("Progressive Tucker warmup must be non-negative")
        self.model = model
        self.optimizer = optimizer
        self.warmup_steps = int(warmup_steps)
        self.seed = int(seed)
        self.verify_rtol = float(verify_rtol)
        self.stages = build_progressive_rank_stages(
            model,
            parse_progressive_stages(stage_values),
        )
        self.stage_index = 0
        self.current_step = 0
        self.last_growth_step: int | None = None
        self.previous_ranks: RankPlan | None = None
        self._hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._record_state()

    def _record_state(self) -> None:
        current = self.stages[self.stage_index]
        self.model._progressive_tucker_state = {
            "version": 1,
            "stage_index": self.stage_index,
            "seed": self.seed,
            "warmup_steps": self.warmup_steps,
            "last_growth_step": self.last_growth_step,
            "module_ranks": {
                name: list(ranks) for name, ranks in current.ranks.items()
            },
            "previous_ranks": (
                None
                if self.previous_ranks is None
                else {
                    name: list(ranks)
                    for name, ranks in self.previous_ranks.items()
                }
            ),
            "stages": [
                {
                    "step": stage.step,
                    "target_parameters": stage.target_parameters,
                    "actual_parameters": stage.actual_parameters,
                }
                for stage in self.stages
            ],
        }

    def _remove_hooks(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _warmup_scale(self) -> float:
        if self.last_growth_step is None or self.warmup_steps == 0:
            return 1.0
        return min(
            1.0,
            max(0.0, (self.current_step - self.last_growth_step) / self.warmup_steps),
        )

    def _install_growth_hooks(self) -> None:
        self._remove_hooks()
        if self.previous_ranks is None or self._warmup_scale() >= 1.0:
            return
        modules = _tucker_modules(self.model)
        for name, module in modules.items():
            old = self.previous_ranks[name]
            new = tuple(module.ranks)
            for factor, old_rank in zip(
                (module.U1, module.U2, module.U3, module.U4), old
            ):
                mask = torch.zeros_like(factor, dtype=torch.bool)
                mask[:, old_rank:] = True

                def factor_hook(gradient, *, growth_mask=mask, controller=self):
                    scale = controller._warmup_scale()
                    return gradient * torch.where(
                        growth_mask,
                        gradient.new_tensor(scale),
                        gradient.new_tensor(1.0),
                    )

                self._hook_handles.append(factor.register_hook(factor_hook))

            old_r1, old_r2, old_r3, old_r4 = old
            new_r1, new_r2, new_r3, new_r4 = new
            core_mask = torch.ones(
                (new_r3, new_r4, new_r1, new_r2),
                device=module.core_matrix.device,
                dtype=torch.bool,
            )
            core_mask[:old_r3, :old_r4, :old_r1, :old_r2] = False
            core_mask = core_mask.reshape_as(module.core_matrix)

            def core_hook(gradient, *, growth_mask=core_mask, controller=self):
                scale = controller._warmup_scale()
                return gradient * torch.where(
                    growth_mask,
                    gradient.new_tensor(scale),
                    gradient.new_tensor(1.0),
                )

            self._hook_handles.append(module.core_matrix.register_hook(core_hook))

    def resume_(self, current_step: int) -> None:
        state = getattr(self.model, "_progressive_tucker_state", None)
        if state is not None:
            self.stage_index = int(state.get("stage_index", 0))
            self.last_growth_step = state.get("last_growth_step")
            raw_previous = state.get("previous_ranks")
            self.previous_ranks = (
                None
                if raw_previous is None
                else {
                    str(name): tuple(int(value) for value in values)
                    for name, values in raw_previous.items()
                }
            )
        self.current_step = int(current_step)
        self._install_growth_hooks()
        self._record_state()

    def maybe_grow(self, current_step: int) -> dict[str, float | int] | None:
        self.current_step = int(current_step)
        if self._hook_handles and self._warmup_scale() >= 1.0:
            self._remove_hooks()

        result = None
        while (
            self.stage_index + 1 < len(self.stages)
            and current_step >= self.stages[self.stage_index + 1].step
        ):
            previous = self.stages[self.stage_index]
            self.stage_index += 1
            stage = self.stages[self.stage_index]
            self.previous_ranks = {
                name: tuple(ranks) for name, ranks in previous.ranks.items()
            }
            self.last_growth_step = stage.step
            result = expand_tucker_model_to_plan_(
                self.model,
                self.optimizer,
                stage.ranks,
                seed=self.seed + self.stage_index * 1_000_003,
                verify_function=True,
                verify_rtol=self.verify_rtol,
            )
            result.update(
                {
                    "stage_index": self.stage_index,
                    "target_parameters": stage.target_parameters,
                    "actual_parameters": stage.actual_parameters,
                    "growth_step": stage.step,
                }
            )
            self._install_growth_hooks()
            self._record_state()
        return result

    def summary_lines(self) -> Iterable[str]:
        for index, stage in enumerate(self.stages):
            yield (
                f"stage {index}: step={stage.step:,}, "
                f"target={stage.target_parameters:,}, "
                f"actual={stage.actual_parameters:,}"
            )


__all__ = [
    "ProgressiveStage",
    "ProgressiveTuckerController",
    "build_progressive_rank_stages",
    "expand_tucker_model_to_plan_",
    "parse_progressive_stages",
    "restore_progressive_tucker_shapes_",
]
