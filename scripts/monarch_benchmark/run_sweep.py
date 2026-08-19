#!/usr/bin/env python3
"""Wait for one uncontended GPU and run the complete large-model sweep on it."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from scripts.monarch_benchmark.common import (
    HARNESS_REVISION,
    MODEL_SPECS,
    VARIANTS,
    atomic_write_json,
    foreign_compute_apps,
    gpu_snapshot,
    query_gpu_inventory,
    result_is_complete,
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


def result_path(output_dir: Path, model_name: str, variant_name: str) -> Path:
    return output_dir / "runs" / f"{model_name}-{variant_name}.json"


def requested_controls(args) -> dict:
    return {
        "harness_revision": HARNESS_REVISION,
        "storage_dtype": "bfloat16",
        "autocast_dtype": "bfloat16",
        "optimizer_moment_dtype": "bfloat16",
        "sequence_length": args.sequence_length,
        "microbatch": args.microbatch,
        "accumulation_steps": args.accumulation_steps,
        "tokens_per_step": args.sequence_length * args.microbatch * args.accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "measured_steps": args.measured_steps,
        "compile_model": False,
        "monarch_blocks": args.monarch_blocks,
        "lr": args.lr,
        "momentum": args.momentum,
        "betas": [args.beta1, args.beta2],
        "weight_decay": args.weight_decay,
        "eps": args.eps,
        "seed": args.seed,
        "contamination_poll_seconds": args.contamination_poll_seconds,
    }


def load_complete(path: Path, gpu_uuid: str, model: dict, variant: dict, args) -> dict | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    return payload if result_matches_request(
        payload,
        gpu_uuid=gpu_uuid,
        model_name=model["name"],
        variant_name=variant["name"],
        controls=requested_controls(args),
    ) else None


def run_worker(gpu: dict, model: dict, variant: dict, args) -> dict:
    output = result_path(args.output_dir, model["name"], variant["name"])
    existing = load_complete(output, gpu["uuid"], model, variant, args)
    if existing is not None and not args.rerun:
        log(f"Reusing {output}")
        return existing

    while True:
        wait_for_selected_gpu(gpu, args)
        command = [
            str(ROOT / ".venv/bin/python"),
            "-m",
            "scripts.monarch_benchmark.benchmark_train_step",
            "--variant",
            variant["name"],
            "--model-size",
            model["name"],
            "--gpu-uuid",
            gpu["uuid"],
            "--sequence-length",
            str(args.sequence_length),
            "--microbatch",
            str(args.microbatch),
            "--accumulation-steps",
            str(args.accumulation_steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--measured-steps",
            str(args.measured_steps),
            "--monarch-blocks",
            str(args.monarch_blocks),
            "--lr",
            str(args.lr),
            "--momentum",
            str(args.momentum),
            "--beta1",
            str(args.beta1),
            "--beta2",
            str(args.beta2),
            "--weight-decay",
            str(args.weight_decay),
            "--eps",
            str(args.eps),
            "--seed",
            str(args.seed),
            "--contamination-poll-seconds",
            str(args.contamination_poll_seconds),
            "--output",
            str(output),
        ]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = str(gpu["index"])
        environment["PYTHONUNBUFFERED"] = "1"
        log(f"Running {model['label']} / {variant['label']} on physical GPU {gpu['index']}")
        completed = subprocess.run(command, cwd=ROOT, env=environment, check=False)
        if not output.is_file():
            raise RuntimeError(f"worker exited {completed.returncode} without producing {output}")
        payload = json.loads(output.read_text())
        if completed.returncode != 0 or not result_is_complete(payload, gpu_uuid=gpu["uuid"]):
            if payload.get("status") == "oom":
                raise RuntimeError(
                    f"OOM for {model['label']} / {variant['label']}; lower sequence length "
                    "and increase accumulation to keep tokens per step fixed"
                )
            if payload.get("status") == "contaminated":
                log(f"Discarding contaminated trial: {payload.get('error')}")
                time.sleep(args.poll_seconds)
                continue
            raise RuntimeError(
                f"worker failed for {model['label']} / {variant['label']}: "
                f"{payload.get('error', payload.get('status'))}"
            )
        foreign = foreign_compute_apps(gpu["uuid"])
        if foreign:
            payload["status"] = "contaminated"
            payload["contamination_after_worker"] = foreign
            atomic_write_json(output, payload)
            log("Foreign process appeared immediately after worker exit; retrying trial")
            time.sleep(args.poll_seconds)
            continue
        return payload


def rotated_variants(model_index: int) -> list[dict]:
    variants = list(VARIANTS)
    offset = model_index % len(variants)
    return variants[offset:] + variants[:offset]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmark_results/monarch-muon-large"))
    parser.add_argument("--gpu-index", type=int, default=None)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-idle-utilization", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--monarch-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--contamination-poll-seconds", type=float, default=0.25)
    parser.add_argument("--rerun", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stable_checks < 1 or args.poll_seconds < 1:
        raise ValueError("stable checks and poll seconds must both be positive")
    if args.contamination_poll_seconds <= 0:
        raise ValueError("contamination poll seconds must be positive")
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gpu, lock_handle = acquire_free_gpu(args)
    log(f"Selected GPU {gpu['index']}: {gpu['name']} ({gpu['uuid']})")
    try:
        results = []
        for model_index, model in enumerate(MODEL_SPECS):
            for variant in rotated_variants(model_index):
                results.append(run_worker(gpu, model, variant, args))
        payload = {
            "status": "complete",
            "gpu": gpu,
            "sweep": {
                "sequence_length": args.sequence_length,
                "microbatch": args.microbatch,
                "accumulation_steps": args.accumulation_steps,
                "tokens_per_step": args.sequence_length * args.microbatch * args.accumulation_steps,
                "warmup_steps": args.warmup_steps,
                "measured_steps": args.measured_steps,
            },
            "results": results,
        }
        atomic_write_json(args.output_dir / "results.json", payload)
        report = ROOT / "reports/monarch-muon-large-benchmark.html"
        subprocess.run(
            [
                str(ROOT / ".venv/bin/python"),
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
        lock_handle.close()


if __name__ == "__main__":
    main()
