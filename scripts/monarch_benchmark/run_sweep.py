#!/usr/bin/env python3
"""Run the complete large-model sweep on one GPU, over models, optimizers and batch sizes."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.monarch_benchmark.common import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TOKENS_PER_STEP,
    MICROBATCHES,
    MODEL_SPECS,
    VARIANTS,
    accumulation_steps,
    atomic_write_json,
    foreign_compute_apps,
    gpu_snapshot,
    query_gpu_inventory,
    requested_controls,
    result_is_recorded,
    result_matches_request,
)

ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def candidate_gpus(requested_index: int | None) -> list[dict]:
    gpus = query_gpu_inventory()
    if requested_index is not None:
        gpus = [gpu for gpu in gpus if gpu["index"] == requested_index]
        if not gpus:
            raise ValueError(f"GPU index {requested_index} is not visible")
    return gpus


def acquire_free_gpu(args):
    locks = {}
    stable_counts = {}
    while True:
        for gpu in candidate_gpus(args.gpu_index):
            uuid = gpu["uuid"]
            if uuid not in locks:
                lock_path = Path("/tmp") / f"effective-muon-benchmark-{uuid}.lock"
                handle = lock_path.open("w")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                locks[uuid] = handle

            foreign = foreign_compute_apps(uuid)
            idle = not foreign and gpu["utilization_gpu_percent"] <= args.max_idle_utilization
            stable_counts[uuid] = stable_counts.get(uuid, 0) + 1 if idle else 0
            if idle:
                log(
                    f"GPU {gpu['index']} {uuid} idle check "
                    f"{stable_counts[uuid]}/{args.stable_checks} "
                    f"(util={gpu['utilization_gpu_percent']}%, mem={gpu['memory_used_mb']} MiB)"
                )
            else:
                log(
                    f"GPU {gpu['index']} unavailable: util={gpu['utilization_gpu_percent']}%, "
                    f"foreign_pids={[process['pid'] for process in foreign]}"
                )
            if stable_counts[uuid] >= args.stable_checks:
                for other_uuid, handle in list(locks.items()):
                    if other_uuid != uuid:
                        handle.close()
                return gpu, locks[uuid]
        time.sleep(args.poll_seconds)


def wait_for_selected_gpu(gpu: dict, args) -> None:
    if args.exclusive_gpu:
        return
    stable = 0
    while stable < args.stable_checks:
        snapshot = gpu_snapshot(gpu["uuid"])
        foreign = foreign_compute_apps(gpu["uuid"])
        idle = not foreign and snapshot["utilization_gpu_percent"] <= args.max_idle_utilization
        stable = stable + 1 if idle else 0
        if not idle:
            log(
                f"Waiting for selected GPU {gpu['index']}: util={snapshot['utilization_gpu_percent']}%, "
                f"foreign_pids={[process['pid'] for process in foreign]}"
            )
        time.sleep(args.poll_seconds if stable < args.stable_checks else 0)


def result_path(output_dir: Path, model_name: str, variant_name: str, microbatch: int) -> Path:
    return output_dir / "runs" / f"{model_name}-{variant_name}-bs{microbatch}.json"


def load_complete(path: Path, gpu_uuid: str | None, model: dict, variant: dict, controls: dict) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload if result_matches_request(
        payload,
        gpu_uuid=gpu_uuid,
        model_name=model["name"],
        variant_name=variant["name"],
        controls=controls,
    ) else None


def worker_command(gpu: dict, model: dict, variant: dict, microbatch: int, accumulation: int, output: Path, args) -> list[str]:
    command = [
        args.python,
        "-m",
        "scripts.monarch_benchmark.benchmark_train_step",
        "--variant", variant["name"],
        "--model-size", model["name"],
        "--sequence-length", str(args.sequence_length),
        "--microbatch", str(microbatch),
        "--accumulation-steps", str(accumulation),
        "--warmup-steps", str(args.warmup_steps),
        "--measured-steps", str(args.measured_steps),
        "--resample-steps", str(args.resample_steps),
        "--monarch-blocks", str(args.monarch_blocks),
        "--density", str(args.density),
        "--update-proj-gap", str(args.update_proj_gap),
        "--lr", str(args.lr),
        "--momentum", str(args.momentum),
        "--beta1", str(args.beta1),
        "--beta2", str(args.beta2),
        "--weight-decay", str(args.weight_decay),
        "--eps", str(args.eps),
        "--seed", str(args.seed),
        "--contamination-poll-seconds", str(args.contamination_poll_seconds),
        "--output", str(output),
    ]
    if args.exclusive_gpu:
        command.append("--exclusive-gpu")
    else:
        command += ["--gpu-uuid", gpu["uuid"]]
    return command


def run_worker(gpu: dict, model: dict, variant: dict, microbatch: int, args) -> dict:
    accumulation = accumulation_steps(args.tokens_per_step, microbatch, args.sequence_length)
    controls = requested_controls(args, microbatch, accumulation)
    output = result_path(args.output_dir, model["name"], variant["name"], microbatch)
    existing = load_complete(output, gpu["uuid"], model, variant, controls)
    if existing is not None and not args.rerun:
        log(f"Reusing {output}")
        return existing

    while True:
        wait_for_selected_gpu(gpu, args)
        command = worker_command(gpu, model, variant, microbatch, accumulation, output, args)
        environment = os.environ.copy()
        if gpu["index"] is not None:
            environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        environment["PYTHONUNBUFFERED"] = "1"
        log(
            f"Running {model['label']} / {variant['label']} / microbatch {microbatch} "
            f"x {accumulation} on GPU {gpu['index']}"
        )
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if not output.is_file():
            raise RuntimeError(f"worker exited {completed.returncode} without producing {output}")
        payload = json.loads(output.read_text())
        if payload.get("status") == "oom":
            # Expected at the top of the batch-size ramp; it is a result, not a failure.
            log(f"Out of memory for {model['label']} / {variant['label']} / microbatch {microbatch}")
            return payload
        if completed.returncode != 0 or not result_is_recorded(payload, gpu_uuid=gpu["uuid"]):
            if payload.get("status") == "contaminated":
                log(f"Discarding contaminated trial: {payload.get('error')}")
                time.sleep(args.poll_seconds)
                continue
            raise RuntimeError(
                f"worker failed for {model['label']} / {variant['label']} / microbatch {microbatch}: "
                f"{payload.get('error', payload.get('status'))}"
            )
        foreign = [] if args.exclusive_gpu else foreign_compute_apps(gpu["uuid"])
        if foreign:
            payload["status"] = "contaminated"
            payload["contamination_after_worker"] = foreign
            atomic_write_json(output, payload)
            log("Foreign process appeared immediately after worker exit; retrying trial")
            time.sleep(args.poll_seconds)
            continue
        return payload


def summary(gpu: dict, results: list[dict], microbatches, models, variants, args) -> dict:
    return {
        "status": "complete",
        "gpu": gpu,
        "sweep": {
            "sequence_length": args.sequence_length,
            "tokens_per_step": args.tokens_per_step,
            "microbatches": sorted(microbatches),
            "models": [model["name"] for model in models],
            "variants": [variant["name"] for variant in variants],
            "warmup_steps": args.warmup_steps,
            "measured_steps": args.measured_steps,
            "resample_steps": args.resample_steps,
            "update_proj_gap": args.update_proj_gap,
            "density": args.density,
            "exclusive_gpu": args.exclusive_gpu,
        },
        "results": results,
    }


def rotated(items: list[dict], offset: int) -> list[dict]:
    position = offset % len(items)
    return items[position:] + items[:position]


def selected(specs, names: str | None, label: str) -> list[dict]:
    if not names:
        return list(specs)
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {spec["name"]: spec for spec in specs}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"unknown {label}: {unknown}")
    return [by_name[name] for name in wanted]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/monarch-muon-large"))
    parser.add_argument("--python", default=sys.executable, help="interpreter used for the workers")
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--exclusive-gpu", action="store_true",
                        help="the GPU is exclusively scheduled, so skip locking and idle checks")
    parser.add_argument("--models", default=None, help="comma-separated subset of model sizes")
    parser.add_argument("--variants", default=None, help="comma-separated subset of variants")
    parser.add_argument("--microbatches", default=None, help="comma-separated subset of microbatch sizes")
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-idle-utilization", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--tokens-per-step", type=int, default=DEFAULT_TOKENS_PER_STEP)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--resample-steps", type=int, default=3)
    parser.add_argument("--monarch-blocks", type=int, default=4)
    parser.add_argument("--density", type=float, default=0.25)
    parser.add_argument("--update-proj-gap", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contamination-poll-seconds", type=float, default=0.25)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stable_checks < 1 or args.poll_seconds < 1:
        raise ValueError("stable checks and poll seconds must both be positive")
    if args.contamination_poll_seconds <= 0:
        raise ValueError("contamination poll seconds must be positive")
    models = selected(MODEL_SPECS, args.models, "model size")
    variants = selected(VARIANTS, args.variants, "variant")
    microbatches = (
        [int(value) for value in args.microbatches.split(",")]
        if args.microbatches
        else list(MICROBATCHES)
    )
    for microbatch in microbatches:
        accumulation_steps(args.tokens_per_step, microbatch, args.sequence_length)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.exclusive_gpu:
        gpu = {"index": args.gpu_index, "uuid": None, "name": "exclusively scheduled"}
        lock_handle = None
        log("Exclusive GPU mode: skipping locking, idle checks and contamination polling")
    else:
        gpu, lock_handle = acquire_free_gpu(args)
        log(f"Selected GPU {gpu['index']}: {gpu['name']} ({gpu['uuid']})")
    try:
        results = []
        # Once a point runs out of memory, every larger microbatch will too.
        exhausted = set()
        for model_index, model in enumerate(models):
            for microbatch in sorted(microbatches):
                for variant in rotated(variants, model_index + microbatch):
                    key = (model["name"], variant["name"])
                    if key in exhausted:
                        log(f"Skipping {model['label']} / {variant['label']} / microbatch "
                            f"{microbatch}: a smaller microbatch already ran out of memory")
                        continue
                    payload = run_worker(gpu, model, variant, microbatch, args)
                    if payload.get("status") == "oom":
                        exhausted.add(key)
                    results.append(payload)
                    # Snapshot after every point so an interrupted sweep still
                    # has a readable matrix to report on.
                    atomic_write_json(args.output_dir / "results.json", summary(gpu, results, microbatches, models, variants, args))
        atomic_write_json(
            args.output_dir / "results.json",
            summary(gpu, results, microbatches, models, variants, args),
        )
        if args.skip_report:
            log(f"Sweep complete: {args.output_dir / 'results.json'}")
            return
        report = ROOT / "reports/memory-efficient-optimizer-benchmark.html"
        subprocess.run(
            [
                args.python,
                "-m",
                "scripts.monarch_benchmark.build_report",
                "--input",
                str(args.output_dir / "results.json"),
                "--output",
                str(report),
            ],
            cwd=ROOT,
            check=True,
        )
        log(f"Sweep complete: {report}")
    finally:
        if lock_handle is not None:
            lock_handle.close()


if __name__ == "__main__":
    main()
