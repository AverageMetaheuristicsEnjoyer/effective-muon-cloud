from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path

from scripts.monarch_benchmark.common import MODEL_SPECS, model_spec


VARIANTS = (
    {"name": "dense_adamw", "label": "Dense AdamW"},
    {"name": "dense_muon", "label": "Dense Muon"},
    {"name": "static_tucker", "label": "Tucker rank-8 iso-param + Tensorion"},
)

MICROBATCHES = (1, 2, 4, 8, 16)
DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_TOKENS_PER_STEP = 16_384
HARNESS_REVISION = 3
TUCKER_RANK_MULTIPLE = 8


def model_geometry(name: str) -> dict:
    spec = model_spec(name)
    intermediate_size = spec["intermediate_size"]
    if not intermediate_size:
        intermediate_size = 256 * (((8 * spec["n_embd"]) // 3 + 255) // 256)
    return {
        "name": spec["name"],
        "label": spec["label"],
        "n_layer": spec["n_layer"],
        "n_embd": spec["n_embd"],
        "n_head": spec["n_head"],
        "intermediate_size": intermediate_size,
        "dense_parameters": spec["dense_params_expected"],
    }


def _aligned_factor_pair(value: int, multiple: int) -> tuple[int, int]:
    left = math.isqrt(value)
    while left and (
        value % left
        or left % multiple
        or (value // left) % multiple
    ):
        left -= 1
    if not left:
        raise ValueError(
            f"feature dimension {value} has no factor pair divisible by {multiple}"
        )
    return left, value // left


def _tucker_parameter_count(
    modes: tuple[int, int, int, int],
    ranks: tuple[int, int, int, int],
) -> int:
    return sum(mode * rank for mode, rank in zip(modes, ranks)) + math.prod(ranks)


def _tucker_module_shapes(geometry: dict) -> list[tuple[str, int, int]]:
    modules = []
    for layer in range(geometry["n_layer"]):
        prefix = f"transformer.h.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            modules.append(
                (f"{prefix}.attn.{projection}", geometry["n_embd"], geometry["n_embd"])
            )
        for projection in ("gate_proj", "up_proj"):
            modules.append(
                (
                    f"{prefix}.mlp.{projection}",
                    geometry["n_embd"],
                    geometry["intermediate_size"],
                )
            )
        modules.append(
            (
                f"{prefix}.mlp.down_proj",
                geometry["intermediate_size"],
                geometry["n_embd"],
            )
        )
    return modules


def tucker_rank_plan(name: str) -> tuple[dict[str, tuple[int, int, int, int]], int]:
    geometry = model_geometry(name)
    modules = _tucker_module_shapes(geometry)
    full_plan = {}
    full_tucker_parameters = 0
    dense_replaced_parameters = 0
    candidates = []

    for module_name, in_features, out_features in modules:
        modes = (
            *_aligned_factor_pair(in_features, TUCKER_RANK_MULTIPLE),
            *_aligned_factor_pair(out_features, TUCKER_RANK_MULTIPLE),
        )
        full_ranks = tuple(modes)
        full_count = _tucker_parameter_count(modes, full_ranks)
        full_plan[module_name] = full_ranks
        full_tucker_parameters += full_count
        dense_replaced_parameters += in_features * out_features

        reductions = []
        for rank_index, rank in enumerate(full_ranks):
            if rank <= TUCKER_RANK_MULTIPLE:
                continue
            reduced = list(full_ranks)
            reduced[rank_index] -= TUCKER_RANK_MULTIPLE
            reductions.append(
                (
                    full_count - _tucker_parameter_count(modes, tuple(reduced)),
                    tuple(reduced),
                )
            )
        reduction, reduced_ranks = min(reductions)
        candidates.append((module_name, reduction, reduced_ranks))

    full_model_parameters = (
        geometry["dense_parameters"]
        - dense_replaced_parameters
        + full_tucker_parameters
    )
    excess = full_model_parameters - geometry["dense_parameters"]
    limit = excess + max(reduction for _, reduction, _ in candidates)
    reachable: dict[int, tuple[int, ...]] = {0: ()}
    for index, (_, reduction, _) in enumerate(candidates):
        additions = {}
        for subtotal, selected in reachable.items():
            new_total = subtotal + reduction
            if new_total <= limit and new_total not in reachable and new_total not in additions:
                additions[new_total] = (*selected, index)
        reachable.update(additions)

    selected_reduction = min(
        reachable,
        key=lambda reduction: (
            abs(full_model_parameters - reduction - geometry["dense_parameters"]),
            full_model_parameters - reduction > geometry["dense_parameters"],
            full_model_parameters - reduction,
        ),
    )
    for index in reachable[selected_reduction]:
        module_name, _, reduced_ranks = candidates[index]
        full_plan[module_name] = reduced_ranks

    actual_parameters = full_model_parameters - selected_reduction
    return full_plan, actual_parameters


def variant_spec(name: str) -> dict:
    for variant in VARIANTS:
        if variant["name"] == name:
            return dict(variant)
    raise KeyError(f"unknown benchmark variant {name!r}")


def accumulation_steps(tokens_per_step: int, microbatch: int, sequence_length: int) -> int:
    tokens_per_microstep = microbatch * sequence_length
    if tokens_per_step % tokens_per_microstep:
        raise ValueError(
            f"{tokens_per_step} tokens per step is not divisible by "
            f"microbatch {microbatch} x sequence length {sequence_length}"
        )
    return tokens_per_step // tokens_per_microstep


def requested_controls(args, microbatch: int, accumulation: int) -> dict:
    _, tucker_parameters = tucker_rank_plan(args.model_size)
    return {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "microbatch": microbatch,
        "accumulation_steps": accumulation,
        "tokens_per_step": args.sequence_length * microbatch * accumulation,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "lr": args.lr,
        "momentum": args.momentum,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "grad_clip": args.grad_clip,
        "seed": args.seed,
        "exclusive_gpu": args.exclusive_gpu,
        "model_size": args.model_size,
        "tucker_rank_multiple": TUCKER_RANK_MULTIPLE,
        "tucker_mode_multiple": TUCKER_RANK_MULTIPLE,
        "tucker_parameters": tucker_parameters,
        "tucker_forward_mode": "contract",
        "static_tucker_component_execution": "five_cuda_streams",
        "tucker_lr_scaling_mode": "first_order_calibrated",
        "tensorion_ns_steps": 6,
        "dense_muon_ns_steps": 5,
        "tucker_vector_transport": True,
    }


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(float(value) for value in values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    numeric = [float(value) for value in values]
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "median": statistics.median(numeric),
        "std": statistics.pstdev(numeric),
        "min": min(numeric),
        "p10": percentile(numeric, 0.1),
        "p90": percentile(numeric, 0.9),
        "max": max(numeric),
    }


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def result_is_recorded(payload: dict) -> bool:
    if payload.get("status") == "oom":
        return True
    requested = payload.get("benchmark", {}).get("measured_steps")
    return (
        payload.get("status") == "complete"
        and bool(payload.get("samples"))
        and requested == len(payload["samples"])
    )


def result_matches_request(payload: dict, variant: str, controls: dict) -> bool:
    if not result_is_recorded(payload):
        return False
    if payload.get("status") == "oom":
        return (
            payload.get("variant") == variant
            and payload.get("requested_controls") == controls
        )
    return (
        payload.get("variant", {}).get("name") == variant
        and all(payload["benchmark"].get(key) == value for key, value in controls.items())
    )
