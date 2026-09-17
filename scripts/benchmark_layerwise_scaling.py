import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


PROFILES = {
    "base": (12, 1024, 8, 2816),
    "deep": (24, 1024, 8, 2816),
    "wide": (12, 2048, 16, 5632),
    "deep-wide": (24, 2048, 16, 5632),
    "large": (32, 2560, 20, 7040),
}
GROUPS = {"small": ("base", "deep", "wide"), "large": ("deep-wide", "large")}
ARMS = (("dense", 0.5), ("A", 0.25), ("B", 0.25), ("A", 0.5), ("B", 0.5))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=GROUPS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--riemannian", action="store_true")
    parser.add_argument("--tucker-implementation", choices=("reference", "grouped"), default="grouped")
    parser.add_argument("--retraction-implementation", choices=("reference", "grouped", "cholesky"), default="grouped")
    args = parser.parse_args()
    cases = []
    for profile in GROUPS[args.group]:
        layers, width, heads, ff = PROFILES[profile]
        for microbatch in (1, 4, 16, 32):
            for variant, fraction in ARMS:
                for kernels in (("native", "liger") if microbatch == 32 else ("native",)):
                    policies = ("adamw",)
                    if args.riemannian:
                        policies = ("reference", "grouped") if variant == "dense" else (args.tucker_implementation,)
                    for policy in policies:
                        name = f"{profile}-{variant}-r{fraction}-mb{microbatch}-{kernels}-{policy}-s42"
                        output = args.output_dir / f"{name}.json"
                        command = [
                            sys.executable, "scripts/benchmark_layerwise_tucker.py",
                            "--layers", str(layers), "--width", str(width), "--heads", str(heads),
                            "--ffn-hidden-size", str(ff), "--variant", variant,
                            "--rank-fraction", str(fraction), "--microbatch", str(microbatch),
                            "--sequence-length", "1024", "--tokens-per-step", "32768",
                            "--warmup", "5", "--steps", "20", "--seed", "42",
                            "--execution", "reordered", "--compile-mode", "max-autotune",
                            "--output", str(output),
                        ]
                        if args.riemannian:
                            command += ["--optimizer", "muon" if variant == "dense" else "riemannian",
                                        "--optimizer-implementation", policy,
                                        "--retraction-implementation", args.retraction_implementation]
                        if microbatch < 32:
                            command.append("--stable-grad-buffers")
                        if kernels == "liger":
                            command.append("--liger")
                        cases.append(dict(name=name, command=command, output=str(output)))
    if args.dry_run:
        print(json.dumps(cases, indent=2))
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.output_dir / f"manifest-{args.group}.json"
    with manifest.open("x") as stream:
        json.dump(dict(status="running", cases=cases), stream, indent=2)
    failed = False
    for case in cases:
        output = Path(case["output"])
        if output.exists():
            raise FileExistsError(output)
        print("BEGIN " + case["name"], flush=True)
        start = time.monotonic()
        try:
            completed = subprocess.run(case["command"], timeout=1200)
            case["returncode"] = completed.returncode
        except subprocess.TimeoutExpired:
            case["returncode"] = 124
        case["wall_seconds"] = time.monotonic() - start
        if output.exists():
            result = json.loads(output.read_text())
            case["status"] = result["status"]
        else:
            case["status"] = "timeout" if case["returncode"] == 124 else "process_failed"
            output.write_text(json.dumps(case, indent=2) + "\n")
        failed |= case["status"] not in ("complete", "oom")
        manifest.write_text(json.dumps(dict(status="running", cases=cases), indent=2) + "\n")
        print("END " + case["name"] + " " + case["status"], flush=True)
    manifest.write_text(json.dumps(dict(status="failed" if failed else "complete", cases=cases), indent=2) + "\n")
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
