from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import tempfile
from pathlib import Path


MODEL_SPECS = (
    {
        "name": "257m",
        "label": "257M",
        "n_layer": 12,
        "n_embd": 1024,
        "n_head": 8,
        "intermediate_size": 0,
        "dense_params_expected": 257_188_864,
        "monarch_params_expected": 163_603_456,
    },
    {
        "name": "834m",
        "label": "834M",
        "n_layer": 24,
        "n_embd": 1536,
        "n_head": 12,
        "intermediate_size": 0,
        "dense_params_expected": 834_086_400,
        "monarch_params_expected": 423_568_896,
    },
    {
        "name": "1p4b",
        "label": "1.44B",
        "n_layer": 24,
        "n_embd": 2048,
        "n_head": 16,
        "intermediate_size": 0,
        "dense_params_expected": 1_439_270_912,
        "monarch_params_expected": 690_587_648,
    },
    {
        "name": "3p5b",
        "label": "3.48B",
        "n_layer": 28,
        "n_embd": 3072,
        "n_head": 24,
        "intermediate_size": 0,
        "dense_params_expected": 3_480_136_704,
        "monarch_params_expected": 1_564_388_352,
    },
    {
        "name": "6p9b",
        "label": "6.89B",
        "n_layer": 32,
        "n_embd": 4096,
        "n_head": 32,
        "intermediate_size": 11008,
        "dense_params_expected": 6_888_361_984,
        "monarch_params_expected": 2_970_882_048,
    },
)

VARIANTS = (
    {"name": "monarch_muon", "label": "Monarch-Muon", "family": "monarch"},
    {"name": "dense_adamw", "label": "Dense AdamW", "family": "dense"},
    {"name": "dense_muon", "label": "Dense Muon", "family": "dense"},
    {"name": "galore", "label": "GaLore", "family": "memory_efficient"},
    {"name": "frugal", "label": "FRUGAL", "family": "memory_efficient"},
    {"name": "apollo", "label": "APOLLO", "family": "memory_efficient"},
    {"name": "apollo_mini", "label": "APOLLO-mini", "family": "memory_efficient"},
    {"name": "fira", "label": "Fira", "family": "memory_efficient"},
)

MICROBATCHES = (1, 2, 4, 8, 16)

DEFAULT_SEQUENCE_LENGTH = 1024
DEFAULT_TOKENS_PER_STEP = 16384

HARNESS_REVISION = 2

# Held identical across every point of a sweep; build_report refuses a mix.
COMMON_CONTROL_FIELDS = (
    "harness_revision",
    "storage_dtype",
    "autocast_dtype",
    "optimizer_moment_dtype",
    "sequence_length",
    "tokens_per_step",
    "warmup_steps",
    "measured_steps",
    "compile_model",
    "monarch_blocks",
    "density",
    "update_proj_gap",
    "resample_steps",
    "lr",
    "momentum",
    "betas",
    "weight_decay",
    "eps",
    "seed",
    "contamination_poll_seconds",
    "exclusive_gpu",
)

# Varied by the sweep, so they identify a point rather than the sweep.
POINT_CONTROL_FIELDS = ("microbatch", "accumulation_steps")


def model_spec(name: str) -> dict:
    for spec in MODEL_SPECS:
        if spec["name"] == name:
            return dict(spec)
    raise KeyError(f"unknown model size {name!r}")


def variant_spec(name: str) -> dict:
    for spec in VARIANTS:
        if spec["name"] == name:
            return dict(spec)
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
    """The control surface a result must reproduce to be reused instead of rerun."""
    return {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "optimizer_moment_dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "microbatch": microbatch,
        "accumulation_steps": accumulation,
        "tokens_per_step": args.sequence_length * microbatch * accumulation,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "compile_model": False,
        "monarch_blocks": args.monarch_blocks,
        "density": args.density,
        "update_proj_gap": args.update_proj_gap,
        "resample_steps": args.resample_steps,
        "lr": args.lr,
        "momentum": args.momentum,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "seed": args.seed,
        "contamination_poll_seconds": args.contamination_poll_seconds,
        "exclusive_gpu": args.exclusive_gpu,
    }


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
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


def parse_gpu_inventory(text: str) -> list[dict]:
    gpus = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 8:
            raise ValueError(f"unexpected nvidia-smi GPU row: {line!r}")
        gpus.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "uuid": fields[2],
                "memory_total_mb": int(fields[3]),
                "memory_used_mb": int(fields[4]),
                "utilization_gpu_percent": int(fields[5]),
                "temperature_c": int(fields[6]),
                "pstate": fields[7],
            }
        )
    return gpus


def query_gpu_inventory() -> list[dict]:
    output = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,temperature.gpu,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return parse_gpu_inventory(output)


def parse_compute_apps(text: str) -> list[dict]:
    processes = []
    for line in text.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",", 3)]
        if len(fields) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {line!r}")
        memory = fields[3]
        processes.append(
            {
                "gpu_uuid": fields[0],
                "pid": int(fields[1]),
                "process_name": fields[2],
                "used_memory_mb": None if memory in ("N/A", "[Not Supported]") else int(memory),
            }
        )
    return processes


def query_compute_apps() -> list[dict]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0 and "No running processes found" not in completed.stderr:
        raise RuntimeError(completed.stderr.strip() or "nvidia-smi process query failed")
    return parse_compute_apps(completed.stdout)


def foreign_compute_apps(gpu_uuid: str, allowed_pids: set[int] | None = None) -> list[dict]:
    allowed = allowed_pids or set()
    return [
        process
        for process in query_compute_apps()
        if process["gpu_uuid"] == gpu_uuid and process["pid"] not in allowed
    ]


def gpu_snapshot(gpu_uuid: str) -> dict:
    for gpu in query_gpu_inventory():
        if gpu["uuid"] == gpu_uuid:
            return gpu
    raise KeyError(f"GPU {gpu_uuid} is no longer visible")


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


def result_is_complete(payload: dict, *, gpu_uuid: str | None = None) -> bool:
    if payload.get("status") != "complete":
        return False
    if gpu_uuid is not None and payload.get("gpu", {}).get("uuid") != gpu_uuid:
        return False
    samples = payload.get("samples", [])
    requested = payload.get("benchmark", {}).get("measured_steps")
    return bool(samples) and requested == len(samples)


def result_is_recorded(payload: dict, *, gpu_uuid: str | None = None) -> bool:
    """A point is finished once it either measured cleanly or ran out of memory."""
    if payload.get("status") == "oom":
        return gpu_uuid is None or payload.get("gpu_uuid") == gpu_uuid
    return result_is_complete(payload, gpu_uuid=gpu_uuid)


def result_matches_request(
    payload: dict,
    *,
    gpu_uuid: str,
    model_name: str,
    variant_name: str,
    controls: dict,
) -> bool:
    if not result_is_recorded(payload, gpu_uuid=gpu_uuid):
        return False
    if payload.get("status") == "oom":
        # An OOM payload carries only the identity and the controls it was asked for.
        return (
            payload.get("model_size") == model_name
            and payload.get("variant") == variant_name
            and payload.get("requested_controls") == controls
        )
    if payload.get("model", {}).get("name") != model_name:
        return False
    if payload.get("variant", {}).get("name") != variant_name:
        return False
    benchmark = payload.get("benchmark", {})
    return all(benchmark.get(key) == value for key, value in controls.items())
