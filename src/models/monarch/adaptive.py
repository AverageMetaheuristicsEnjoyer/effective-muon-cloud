"""Function-preserving transitions between progressively denser Monarch layers."""
from __future__ import annotations

import gc
import math
from collections.abc import Sequence

import torch
import torch.nn as nn

from .monarch_linear import MonarchLinear


def validate_schedule(blocks: Sequence[int], fractions: Sequence[float]) -> None:
    """Validate a strictly decreasing block schedule ending in a dense layer."""
    if len(blocks) < 2 or len(blocks) != len(fractions):
        raise ValueError("blocks and fractions must have the same length >= 2")
    if blocks[-1] != 1:
        raise ValueError("the final block count must be 1 (a conventional dense layer)")
    if any(block <= 0 for block in blocks):
        raise ValueError("block counts must be positive")
    if any(old <= new or old % new for old, new in zip(blocks, blocks[1:])):
        raise ValueError("each block count must be a strict divisor of the previous one")
    if any(fraction <= 0 for fraction in fractions):
        raise ValueError("stage fractions must be positive")
    if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-8):
        raise ValueError("stage fractions must sum to 1")


def transition_steps(iterations: int, fractions: Sequence[float]) -> tuple[int, ...]:
    """Return completed-step boundaries at which the next stage becomes active."""
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    boundaries = []
    cumulative = 0.0
    for fraction in fractions[:-1]:
        cumulative += fraction
        boundaries.append(max(1, min(iterations - 1, round(iterations * cumulative))))
    if len(set(boundaries)) != len(boundaries):
        raise ValueError("iterations are too few for the requested stage fractions")
    return tuple(boundaries)


def stage_index_for_step(step: int, boundaries: Sequence[int]) -> int:
    return sum(step >= boundary for boundary in boundaries)


def _copy_bias(source: nn.Module, target: nn.Module) -> None:
    if source.bias is not None:
        target.bias.copy_(source.bias)


@torch.no_grad()
def embed_monarch(source: MonarchLinear, target: MonarchLinear) -> None:
    """Embed source exactly into a lower-block-count Monarch target."""
    old_blocks = source.blkdiag1.shape[0]
    new_blocks = target.blkdiag1.shape[0]
    if old_blocks <= new_blocks or old_blocks % new_blocks:
        raise ValueError(f"cannot embed {old_blocks} blocks into {new_blocks} blocks")
    if (
        source.in_features_extended != target.in_features_extended
        or source.out_features_extended != target.out_features_extended
    ):
        raise ValueError(
            "exact Monarch embedding requires dimensions divisible by both block counts"
        )

    old_q, old_p = source.blkdiag1.shape[1:]
    old_s, old_r = source.blkdiag2.shape[1:]
    new_q, new_p = target.blkdiag1.shape[1:]
    new_s, new_r = target.blkdiag2.shape[1:]
    ratio = old_blocks // new_blocks
    if not (
        new_p == ratio * old_p
        and new_q == ratio * old_q
        and new_r == ratio * old_r
        and new_s == ratio * old_s
    ):
        raise ValueError("unsupported Monarch factor shapes for exact embedding")

    # The butterfly riffle changes when the number of blocks changes. Build a
    # permutation of the first-factor coordinates that keeps each old second
    # factor inside one new second-factor block.
    coordinate_map = [-1] * (old_blocks * old_q)
    for new_first_block in range(new_blocks):
        sources = range(
            new_first_block * new_q,
            (new_first_block + 1) * new_q,
        )
        for new_second_block in range(new_blocks):
            matching_sources = [
                coordinate
                for coordinate in sources
                if (coordinate % old_blocks) % new_blocks == new_second_block
            ]
            targets = [
                coordinate
                for coordinate in sources
                if coordinate % new_blocks == new_second_block
            ]
            if len(matching_sources) != len(targets):
                raise RuntimeError("failed to construct a Monarch coordinate embedding")
            for source_coordinate, target_coordinate in zip(matching_sources, targets):
                coordinate_map[source_coordinate] = target_coordinate

    target.blkdiag1.zero_()
    for old_first_block in range(old_blocks):
        new_first_block = old_first_block // ratio
        input_offset = (old_first_block % ratio) * old_p
        for old_row in range(old_q):
            source_coordinate = old_first_block * old_q + old_row
            target_coordinate = coordinate_map[source_coordinate]
            if target_coordinate // new_q != new_first_block:
                raise RuntimeError("coordinate embedding crossed a first-factor block")
            new_row = target_coordinate % new_q
            target.blkdiag1[
                new_first_block,
                new_row,
                input_offset : input_offset + old_p,
            ].copy_(source.blkdiag1[old_first_block, old_row])

    target.blkdiag2.zero_()
    for old_second_block in range(old_blocks):
        new_second_block = old_second_block % new_blocks
        output_phase = old_second_block // new_blocks
        for old_input in range(old_r):
            # Invert the old riffle, apply the coordinate map, then apply the
            # new riffle to locate this coordinate in the target factor.
            source_coordinate = old_input * old_blocks + old_second_block
            target_coordinate = coordinate_map[source_coordinate]
            if target_coordinate % new_blocks != new_second_block:
                raise RuntimeError("coordinate embedding crossed a second-factor block")
            new_input = target_coordinate // new_blocks
            target.blkdiag2[
                new_second_block,
                output_phase::ratio,
                new_input,
            ].copy_(source.blkdiag2[old_second_block, :, old_input])

    _copy_bias(source, target)


@torch.no_grad()
def materialize_dense(source: MonarchLinear, target: nn.Linear) -> None:
    """Materialize a Monarch layer into a conventional dense Linear exactly."""
    if (
        source.in_features_extended != source.in_features
        or source.out_features_extended != source.out_features
    ):
        raise ValueError("dense materialization currently requires unpadded dimensions")
    identity = torch.eye(
        source.in_features,
        device=source.blkdiag1.device,
        dtype=source.blkdiag1.dtype,
    )
    # TF32 is appropriate for training throughput, but not for computing the
    # single FP32 matrix that replaces a product of two factors.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        if identity.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
        # Linear stores [out, in], while basis rows produce [in, out].
        target.weight.copy_(source.forward_matmul(identity).transpose(0, 1))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    _copy_bias(source, target)


class AdaptiveMonarchLinear(nn.Module):
    """A linear layer whose registered stages become progressively denser.

    All stages are registered before DDP construction. Only the active stage is
    executed; this keeps checkpointing and distributed parameter topology stable.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        blocks: Sequence[int],
        *,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.blocks = tuple(blocks)
        stages: list[nn.Module] = []
        for block_count in self.blocks:
            if block_count == 1:
                stage = nn.Linear(
                    in_features,
                    out_features,
                    bias=bias,
                    device=device,
                    dtype=dtype,
                )
            else:
                stage = MonarchLinear(
                    in_features,
                    out_features,
                    bias=bias,
                    nblocks=block_count,
                    device=device,
                    dtype=dtype,
                )
            stages.append(stage)
        self.stages = nn.ModuleList(stages)
        self.active_stage_index = 0

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        blocks: Sequence[int],
    ) -> "AdaptiveMonarchLinear":
        return cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            blocks,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

    @property
    def active_stage(self) -> nn.Module:
        return self.stages[self.active_stage_index]

    @property
    def active_blocks(self) -> int:
        return self.blocks[self.active_stage_index]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.active_stage(inputs)

    @torch.no_grad()
    def transition_to(self, target_index: int, verify: bool = True) -> float:
        if target_index == self.active_stage_index:
            return 0.0
        if target_index != self.active_stage_index + 1:
            raise ValueError("adaptive transitions must advance exactly one stage")

        source = self.active_stage
        target = self.stages[target_index]
        if isinstance(target, MonarchLinear):
            embed_monarch(source, target)
        elif isinstance(source, MonarchLinear) and isinstance(target, nn.Linear):
            materialize_dense(source, target)
        else:
            raise TypeError("unsupported adaptive Monarch transition")

        max_error = 0.0
        if verify:
            parameter = next(source.parameters())
            device = source.bias.device if source.bias is not None else parameter.device
            generator = torch.Generator(device=device)
            generator.manual_seed(17)
            probe = torch.randn(
                4,
                self.in_features,
                generator=generator,
                device=device,
                dtype=parameter.dtype,
            )
            previous_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                if probe.is_cuda:
                    torch.backends.cuda.matmul.allow_tf32 = False
                source_output = source(probe)
                target_output = target(probe)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = previous_tf32
            max_error = (source_output - target_output).abs().max().item()
            scale = source_output.abs().max().item()
            tolerance = 2e-5 * max(1.0, scale)
            if max_error > tolerance:
                raise RuntimeError(
                    f"Monarch transition changed the function: {max_error:.3e} > {tolerance:.3e}"
                )

        self.active_stage_index = target_index
        return max_error

    def get_extra_state(self) -> dict[str, int]:
        return {"active_stage_index": self.active_stage_index}

    def set_extra_state(self, state: dict[str, int]) -> None:
        index = int(state["active_stage_index"])
        if not 0 <= index < len(self.stages):
            raise ValueError(f"invalid active stage index {index}")
        self.active_stage_index = index


_DEFAULT_EXCLUDE = ("lm_head",)


def apply_adaptive_monarch(
    model: nn.Module,
    blocks: Sequence[int],
    exclude: tuple[str, ...] = _DEFAULT_EXCLUDE,
    verbose: bool = True,
) -> int:
    """Replace eligible Linear modules with AdaptiveMonarchLinear in place."""
    replacements = []
    for parent_name, parent in model.named_modules():
        if isinstance(parent, AdaptiveMonarchLinear):
            continue
        for child_name, child in parent.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if any(excluded in full_name for excluded in exclude):
                continue
            for block_count in blocks:
                if (
                    child.in_features % block_count
                    or child.out_features % block_count
                ):
                    raise ValueError(
                        f"{full_name} shape {tuple(child.weight.shape)} is not divisible "
                        f"by block count {block_count}; exact transitions require divisibility"
                    )
            replacement = AdaptiveMonarchLinear.from_linear(child, blocks)
            replacements.append((parent, child_name, replacement, full_name))

    for parent, child_name, replacement, full_name in replacements:
        setattr(parent, child_name, replacement)
        if verbose:
            print(
                f"  Adaptive Monarch: {full_name} "
                f"{replacement.in_features}->{replacement.out_features}, "
                f"blocks={list(blocks)}"
            )
    gc.collect()
    return len(replacements)


def adaptive_modules(model: nn.Module) -> list[AdaptiveMonarchLinear]:
    return [module for module in model.modules() if isinstance(module, AdaptiveMonarchLinear)]


@torch.no_grad()
def transition_adaptive_model(
    model: nn.Module,
    target_index: int,
    verify: bool = True,
) -> float:
    errors = [
        module.transition_to(target_index, verify=verify)
        for module in adaptive_modules(model)
    ]
    return max(errors, default=0.0)


def active_parameter_count(model: nn.Module) -> int:
    adaptive_parameter_ids = {
        id(parameter)
        for module in adaptive_modules(model)
        for parameter in module.parameters()
    }
    count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in adaptive_parameter_ids
    )
    count += sum(
        parameter.numel()
        for module in adaptive_modules(model)
        for parameter in module.active_stage.parameters()
    )
    return count
