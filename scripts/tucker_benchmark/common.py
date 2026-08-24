from __future__ import annotations

import json
import math
import os
import statistics
import tempfile
from pathlib import Path


VARIANTS = (
    {"name": "dense_adamw", "label": "Dense AdamW", "parameters": 257_188_864},
    {"name": "dense_muon", "label": "Dense Muon", "parameters": 257_188_864},
    {
        "name": "static_tucker",
        "label": "Static Tucker + Tensorion",
        "parameters": 257_676_352,
    },
)

MICROBATCHES = (1, 2, 4, 8, 16)
DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_TOKENS_PER_STEP = 16_384
HARNESS_REVISION = 2


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
        "tucker_rank": 259,
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
