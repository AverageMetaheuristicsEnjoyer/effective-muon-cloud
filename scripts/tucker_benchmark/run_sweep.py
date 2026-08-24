#!/usr/bin/env python3
"""Run the resumable static-Tucker benchmark sweep on one exclusive GPU."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from scripts.tucker_benchmark.common import (
    DEFAULT_SEQUENCE_LENGTH,
    DEFAULT_TOKENS_PER_STEP,
    MICROBATCHES,
    VARIANTS,
    accumulation_steps,
    atomic_write_json,
    requested_controls,
    result_matches_request,
)

ROOT = Path(__file__).resolve().parents[2]


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())} UTC] {message}", flush=True)


def selected_variants(names: str | None) -> list[dict]:
    if not names:
        return list(VARIANTS)
    wanted = [name.strip() for name in names.split(",") if name.strip()]
    by_name = {variant["name"]: variant for variant in VARIANTS}
    unknown = [name for name in wanted if name not in by_name]
    if unknown:
        raise ValueError(f"unknown variants: {unknown}")
    return [by_name[name] for name in wanted]


def result_path(output_dir: Path, variant: str, microbatch: int) -> Path:
    return output_dir / "runs" / f"llama-257m-{variant}-bs{microbatch}.json"


def run_point(args, variant: dict, microbatch: int) -> dict:
    accumulation = accumulation_steps(
        args.tokens_per_step, microbatch, args.sequence_length
    )
    controls = requested_controls(args, microbatch, accumulation)
    output = result_path(args.output_dir, variant["name"], microbatch)
    if output.is_file() and not args.rerun:
        payload = json.loads(output.read_text())
        if result_matches_request(payload, variant["name"], controls):
            log(f"Reusing {output}")
            return payload

    command = [
        args.python,
        "-m",
        "scripts.tucker_benchmark.benchmark_train_step",
        "--variant",
        variant["name"],
        "--exclusive-gpu",
        "--sequence-length",
        str(args.sequence_length),
        "--microbatch",
        str(microbatch),
        "--accumulation-steps",
        str(accumulation),
        "--warmup-steps",
        str(args.warmup_steps),
        "--measured-steps",
        str(args.measured_steps),
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
        "--grad-clip",
        str(args.grad_clip),
        "--seed",
        str(args.seed),
        "--output",
        str(output),
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    log(
        f"Running {variant['label']} / microbatch {microbatch} x {accumulation}"
    )
    completed = subprocess.run(
        command, cwd=ROOT, env=environment, check=False
    )
    if not output.is_file():
        raise RuntimeError(
            f"worker exited {completed.returncode} without producing {output}"
        )
    payload = json.loads(output.read_text())
    if payload.get("status") == "oom":
        log(f"Out of memory: {variant['label']} / microbatch {microbatch}")
        return payload
    if completed.returncode != 0 or payload.get("status") != "complete":
        raise RuntimeError(
            f"worker failed for {variant['label']} / microbatch {microbatch}: "
            f"{payload.get('error', payload.get('status'))}"
        )
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmark_results/static-tucker-257m"),
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--variants", default=None)
    parser.add_argument("--microbatches", default=None)
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument("--tokens-per-step", type=int, default=DEFAULT_TOKENS_PER_STEP)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measured-steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--momentum", type=float, default=0.95)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.99)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--exclusive-gpu", action="store_true", default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    variants = selected_variants(args.variants)
    microbatches = (
        [int(value) for value in args.microbatches.split(",")]
        if args.microbatches
        else list(MICROBATCHES)
    )
    for microbatch in microbatches:
        accumulation_steps(args.tokens_per_step, microbatch, args.sequence_length)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    exhausted = set()
    for microbatch in sorted(microbatches):
        for variant in variants:
            if variant["name"] in exhausted:
                continue
            payload = run_point(args, variant, microbatch)
            results.append(payload)
            if payload.get("status") == "oom":
                exhausted.add(variant["name"])
            atomic_write_json(
                args.output_dir / "results.json",
                {
                    "status": "running",
                    "variants": [item["name"] for item in variants],
                    "microbatches": sorted(microbatches),
                    "results": results,
                },
            )

    atomic_write_json(
        args.output_dir / "results.json",
        {
            "status": "complete",
            "variants": [item["name"] for item in variants],
            "microbatches": sorted(microbatches),
            "results": results,
        },
    )
    log(f"Sweep complete: {args.output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
