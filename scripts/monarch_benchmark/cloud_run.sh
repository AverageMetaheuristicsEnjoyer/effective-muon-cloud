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
