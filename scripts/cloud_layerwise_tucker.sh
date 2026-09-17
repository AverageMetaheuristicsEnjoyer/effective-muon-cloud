#!/bin/bash
set -uo pipefail

RESULTS=${LAYERWISE_TUCKER_RESULTS:-/workspace-SR006.nfs3/layerwise-tucker-20260917}
MODE=${1:-sweep}
if [ "$MODE" = export ]; then
    python3 - "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob("runs/*.json")):
    print("RESULT_JSON " + json.dumps({"file": path.name, "result": json.loads(path.read_text())}))
PY
    exit $?
fi

mkdir -p "$RESULTS/logs" "$RESULTS/runs"
LOG="$RESULTS/logs/${MODE}-$(date -u +%Y%m%dT%H%M%S)-$$.log"
run() {
    export PYTHONNOUSERSITE=1
    export PYTHONUNBUFFERED=1
    export OMP_NUM_THREADS=4
    export PYTHONPATH="/tmp/layerwise-tucker-deps:src:."
    git rev-parse HEAD
    df -h "$RESULTS"
    df -i "$RESULTS"
    python -m pip install --disable-pip-version-check --target /tmp/layerwise-tucker-deps tiktoken || return $?
    python -m unittest discover -s tests -p test_layerwise_tucker.py -v || return $?
    if [ "$MODE" = selftest ]; then
        for variant in dense A B; do
            python scripts/benchmark_layerwise_tucker.py --device cpu --tiny \
                --variant "$variant" --sequence-length 8 --microbatch 2 \
                --tokens-per-step 32 --warmup 1 --steps 2 \
                --output "$RESULTS/smoke-${variant}.json" || return $?
        done
        return 0
    fi
    nvidia-smi || return $?
    status=0
    for repeat in 0 1 2; do
        seed=$((42 + repeat))
        arms=(dense:0.5 A:0.25 B:0.25 A:0.5 B:0.5)
        for mb in 1 16; do
            for offset in 0 1 2 3 4; do
                arm=${arms[$(((offset + repeat * 2) % 5))]}
                variant=${arm%:*}
                fraction=${arm#*:}
                name="${variant}-r${fraction}-mb${mb}-seed${seed}"
                echo "BEGIN $name"
                python scripts/benchmark_layerwise_tucker.py \
                    --variant "$variant" --rank-fraction "$fraction" \
                    --microbatch "$mb" --seed "$seed" \
                    --output "$RESULTS/runs/${name}.json" || status=1
                echo "END $name"
            done
        done
    done
    return "$status"
}
run 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "APPLICATION_EXIT=$status"
echo "LOG=$LOG"
printf '%s\n' "$status" > "$RESULTS/${MODE}.exit"
# The gateway hides failed-job logs; APPLICATION_EXIT is the success criterion.
exit 0
