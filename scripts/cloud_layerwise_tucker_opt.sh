#!/bin/bash
set -uo pipefail
RESULTS=${LAYERWISE_OPT_RESULTS:-/workspace-SR006.nfs3/layerwise-tucker-opt-20260917}
MODE=${1:-screen}
if [ "$MODE" = export ]; then
    python3 - "$RESULTS" "${2:-screen}" <<'PY'
import json
import sys
from pathlib import Path
for path in sorted((Path(sys.argv[1]) / sys.argv[2]).glob('*.json')):
    print('RESULT_JSON ' + json.dumps({'file': path.name, 'result': json.loads(path.read_text())}))
PY
    exit $?
fi
if [ "$MODE" = peek ]; then
    python3 - "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
for path in sorted(root.glob('*/*.json')):
    r = json.loads(path.read_text())
    print(str(path.relative_to(root)), r.get('status'), r.get('summary', {}).get('host_step_ms', {}).get('median'), r.get('error', '')[-600:])
for path in sorted(root.glob('logs/*.log')):
    print(str(path), '\n'.join(path.read_text().splitlines()[-8:]))
PY
    exit $?
fi
mkdir -p "$RESULTS/logs" "$RESULTS/$MODE"
LOG="$RESULTS/logs/${MODE}-$(date -u +%Y%m%dT%H%M%S)-$$.log"
run() {
    export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4
    export PYTHONPATH="/tmp/layerwise-opt-deps:src:."
    export TORCHINDUCTOR_COMPILE_THREADS=4
    export TORCHINDUCTOR_CACHE_DIR="$RESULTS/inductor-cache"
    export TRITON_CACHE_DIR="$RESULTS/triton-cache"
    git rev-parse HEAD
    df -h "$RESULTS"
    df -i "$RESULTS"
    python -m pip install --disable-pip-version-check --target /tmp/layerwise-opt-deps tiktoken || return $?
    python -m pip install --disable-pip-version-check --no-deps --target /tmp/layerwise-opt-deps liger-kernel==0.8.1 || return $?
    python -m unittest discover -s tests -p test_layerwise_tucker.py -v || return $?
    if [ "$MODE" = selftest ]; then
        for v in dense A B; do
            python scripts/benchmark_layerwise_tucker.py --device cpu --tiny --variant "$v" \
                --execution reordered --sequence-length 8 --microbatch 2 --tokens-per-step 32 \
                --warmup 1 --steps 2 --output "$RESULTS/$MODE/$v.json" || return $?
        done
        return 0
    fi
    nvidia-smi || return $?
    status=0
    if [ "$MODE" = screen ]; then
        python scripts/validate_layerwise_tucker_opt.py --output "$RESULTS/screen-correctness.json" || return $?
        for mb in 1 16; do
            for arm in dense:0.5 A:0.25 B:0.25 A:0.5 B:0.5; do
                variant=${arm%:*}; fraction=${arm#*:}
                for execution in reference reordered; do
                    [ "$variant" = dense ] && [ "$execution" = reordered ] && continue
                    name="$variant-r$fraction-mb$mb-$execution"
                    extra=()
                    if [ "$mb" = 16 ] && { [ "$variant" = dense ] || { [ "$variant" = B ] && [ "$fraction" = 0.25 ]; }; }; then
                        extra=(--profile)
                    fi
                    echo "BEGIN $name"
                    python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                        --microbatch "$mb" --execution "$execution" "${extra[@]}" \
                        --output "$RESULTS/$MODE/$name.json" || status=1
                    echo "END $name"
                done
            done
        done
    elif [ "$MODE" = compile ]; then
        for compiler in reduce-overhead max-autotune; do
            python scripts/validate_layerwise_tucker_opt.py --compile-mode "$compiler" \
                --output "$RESULTS/compile-$compiler-correctness.json" || return $?
        done
        for mb in 1 16; do
            for arm in dense:0.5 A:0.5 B:0.25; do
                variant=${arm%:*}; fraction=${arm#*:}
                for compiler in reduce-overhead max-autotune; do
                    name="$variant-r$fraction-mb$mb-$compiler"
                    echo "BEGIN $name"
                    timeout 900 python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                        --microbatch "$mb" --execution reordered --compile-mode "$compiler" \
                        --output "$RESULTS/$MODE/$name.json" || status=1
                    echo "END $name"
                done
            done
        done
    elif [ "$MODE" = liger ]; then
        python scripts/validate_layerwise_tucker_opt.py --liger --output "$RESULTS/liger-correctness.json" || return $?
        for mb in 1 16; do
            for arm in dense:0.5 A:0.25 B:0.25 A:0.5 B:0.5; do
                variant=${arm%:*}; fraction=${arm#*:}
                name="$variant-r$fraction-mb$mb-liger"
                echo "BEGIN $name"
                python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                    --microbatch "$mb" --execution reordered --liger \
                    --output "$RESULTS/$MODE/$name.json" || status=1
                echo "END $name"
            done
        done
    else
        echo "Unknown mode: $MODE"
        return 2
    fi
    return "$status"
}
run 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "APPLICATION_EXIT=$status"
echo "LOG=$LOG"
printf '%s\n' "$status" > "$RESULTS/$MODE.exit"
exit 0
