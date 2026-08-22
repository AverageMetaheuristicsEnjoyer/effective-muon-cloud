#!/bin/bash
# Entry point for cloud.ru jobs launched through mlsub.
#
#   mlsub run --repo <public mirror> --branch <branch> --image torch28 --no-pip \
#     --entry scripts/monarch_benchmark/cloud_run.sh --gpus 1 \
#     --args "--models 257m,834m --variants dense_adamw,galore,frugal,apollo,apollo_mini,fira"
#
# A failed mlsub job shows no logs at all, so everything is captured to the
# persistent workspace disk and the tail is echoed before exiting zero.
set -u

RESULTS=${MEMBENCH_RESULTS:-/home/jovyan/mem-eff-bench}
mkdir -p "$RESULTS/logs" "$RESULTS/results"
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
} >"$LOG" 2>&1
status=$?

echo "EXIT=$status"
echo "log: $LOG"
echo "=== last 120 lines ==="
tail -120 "$LOG"
exit 0
