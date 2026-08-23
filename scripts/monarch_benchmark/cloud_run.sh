#!/bin/bash
# Entry point for cloud.ru jobs launched through mlsub.
#
#   mlsub run --repo <public mirror> --branch <branch> --image torch28 --no-pip \
#     --entry scripts/monarch_benchmark/cloud_run.sh --gpus 1 \
#     --args "--variants dense_adamw,galore,frugal,apollo,apollo_mini,fira"
#
# First argument may instead be:
#   selftest   run the unit tests (the cheap --gpus cpu rehearsal)
#   peek       print the newest log and the results recorded so far
#   export     print one compact line per recorded point, for mlsub logs
#
# A failed mlsub job shows no logs at all, so output is teed to the persistent
# workspace disk and this script always exits zero.
set -u

RESULTS=${MEMBENCH_RESULTS:-/home/jovyan/mem-eff-bench}
mkdir -p "$RESULTS/logs" "$RESULTS/results"

if [ "${1:-}" = "peek" ]; then
    echo "=== recorded points ==="
    ls -1 "$RESULTS/results/runs" 2>/dev/null | sort || echo "none yet"
    newest=$(ls -t "$RESULTS"/logs/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-120}" "$newest"
    exit 0
fi

if [ "${1:-}" = "export" ]; then
    # mlsub logs only keeps the tail, so every point leaves as one compact line
    # rather than as its full JSON payload.
    python3 - "$RESULTS/results/runs" <<'PY'
import json
import sys
from pathlib import Path

COLUMNS = ("model", "variant", "microbatch", "status", "median_ms", "optimizer_ms",
           "tokens_per_second", "peak_gb", "state_gb", "projector_gb", "params_gb",
           "resample_ms", "resample_peak_gb")


def cell(value, digits):
    return "" if value is None else f"{value:.{digits}f}"


print("PT\t" + "\t".join(COLUMNS))
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    model, variant, batch = path.stem.split("-")
    payload = json.loads(path.read_text())
    status = payload.get("status")
    row = [model, variant, batch.removeprefix("bs"), status]
    if status == "complete":
        summary = payload["summary"]
        memory = payload["memory"]
        resample = payload.get("resample_summary")
        resample_memory = payload.get("resample_memory")
        row += [
            cell(summary["host_total_ms"]["median"], 3),
            cell(summary["optimizer_ms"]["median"], 3),
            cell(summary["tokens_per_second"]["median"], 1),
            cell(memory["peak_allocated_bytes"] / 1e9, 3),
            cell(memory["optimizer_state_bytes"] / 1e9, 3),
            cell(memory["optimizer_projector_bytes"] / 1e9, 3),
            cell(memory["model_bytes"] / 1e9, 3),
            cell(resample["host_total_ms"]["median"] if resample else None, 3),
            cell(resample_memory["peak_allocated_bytes"] / 1e9 if resample_memory else None, 3),
        ]
    else:
        row += [""] * (len(COLUMNS) - len(row))
    print("PT\t" + "\t".join(str(field) for field in row))
PY
    exit 0
fi

if [ "${1:-}" = "relocate" ]; then
    # /home/jovyan reached its 500k inode quota, so a sweep has to be able to
    # carry its recorded points to a volume that still has file slots.
    destination=$2
    mkdir -p "$destination"
    cp -a "$RESULTS/results" "$destination/"
    echo "points at $destination: $(ls -1 "$destination/results/runs" 2>/dev/null | wc -l)"
    exit 0
fi

if [ "${1:-}" = "disk" ]; then
    # ENOSPC in the results directory has outlived a df that reported free
    # space, so report the file quota and probe every volume for a write.
    df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    echo "recorded points: $(ls -1 "$RESULTS/results/runs" 2>/dev/null | wc -l)"
    for target in "$RESULTS/results" /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
        if probe=$(mktemp "$target/.membench-probe.XXXXXX" 2>&1); then
            echo "writable: $target"
            rm -f "$probe"
        else
            echo "NOT writable: $target ($probe)"
        fi
    done
    exit 0
fi

LOG="$RESULTS/logs/$(date +%F_%H%M%S)-$$.log"

{
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python: $(python -V 2>&1)"
    nvidia-smi || echo "nvidia-smi unavailable"

    # The image ships torch only, and requirements.txt pulls in the whole
    # training stack; these three are what the benchmark actually imports.
    for package in tiktoken transformers bitsandbytes; do
        python -c "import $package" 2>/dev/null || pip install --user -q "$package"
    done

    export PYTHONPATH=.:src
    if [ "${1:-}" = "selftest" ]; then
        shift
        python -m unittest tests.test_monarch_large_benchmark
    else
        python -m scripts.monarch_benchmark.run_sweep \
            --exclusive-gpu \
            --output-dir "$RESULTS/results" \
            --skip-report \
            "$@"
    fi
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo "EXIT=$status"
echo "log: $LOG"
exit 0
