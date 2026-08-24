#!/bin/bash
# Cloud.ru mlsub entry point. It keeps logs and results on the persistent nfs3 volume.
set -u

RESULTS=${TUCKER_BENCH_RESULTS:-/workspace-SR006.nfs3/tucker-membench}
mkdir -p "$RESULTS/logs" "$RESULTS/results"

if [ "${1:-}" = "peek" ]; then
    newest=$(ls -t "$RESULTS"/logs/*.log 2>/dev/null | head -1)
    echo "points: $(ls -1 "$RESULTS/results/runs" 2>/dev/null | wc -l)"
    echo "latest log: ${newest:-none}"
    [ -n "$newest" ] && tail -"${2:-160}" "$newest"
    exit 0
fi

if [ "${1:-}" = "export" ]; then
    python3 - "$RESULTS/results/runs" <<'PY'
import json
import sys
from pathlib import Path

print("variant\tmicrobatch\tstatus\tmedian_ms\ttokens_per_second\tpeak_gb\tstate_gb\tmodel_gb\toptimizer_ms\tgrad_clip_ms\tgpu")
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    payload = json.loads(path.read_text())
    status = payload.get("status")
    if status != "complete":
        print(f"{payload.get('variant', '')}\t{payload.get('requested_controls', {}).get('microbatch', '')}\t{status}")
        continue
    summary = payload["summary"]
    memory = payload["memory"]
    print("\t".join(map(str, (
        payload["variant"]["name"],
        payload["benchmark"]["microbatch"],
        status,
        round(summary["host_total_ms"]["median"], 3),
        round(summary["tokens_per_second"]["median"], 1),
        round(memory["peak_allocated_bytes"] / 1e9, 3),
        round(memory["optimizer_state_bytes"] / 1e9, 3),
        round(memory["model_bytes"] / 1e9, 3),
        round(summary["optimizer_ms"]["median"], 3),
        round(summary["grad_clip_ms"]["median"], 3),
        payload["gpu"]["name"],
    ))))
PY
    exit 0
fi

LOG="$RESULTS/logs/$(date +%F_%H%M%S)-$$.log"
{
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python: $(python -V 2>&1)"
    export PYTHONPATH=.:src
    if ! python -c "import tiktoken" 2>/dev/null; then
        mkdir -p "$RESULTS/python"
        pip install --target "$RESULTS/python" -q tiktoken
        export PYTHONPATH="$RESULTS/python:$PYTHONPATH"
    fi
    if [ "${1:-}" = "selftest" ]; then
        python -m unittest tests.test_tucker_benchmark
    else
        python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/results" \
            "$@"
    fi
} 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}

echo "EXIT=$status"
echo "log: $LOG"
exit 0
