#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.tucker_benchmark.common import tucker_benchmark_plan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layout",
        required=True,
        choices=("order3_input", "order3_output", "order3_paired"),
    )
    parser.add_argument("--profile", default="progressive_225m_rank8")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    _, plan, parameters, profile = tucker_benchmark_plan(
        "257m", args.profile, args.layout
    )
    payload = {
        "model": "257m",
        "profile": profile,
        "actual_parameters": parameters,
        "module_ranks": {name: list(ranks) for name, ranks in plan.items()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        f"layout={args.layout} profile={args.profile} "
        f"parameters={parameters} output={args.output}"
    )


if __name__ == "__main__":
    main()
