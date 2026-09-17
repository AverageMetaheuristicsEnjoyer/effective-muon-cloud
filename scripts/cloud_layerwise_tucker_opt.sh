#!/bin/bash
set -uo pipefail
RESULTS=${LAYERWISE_OPT_RESULTS:-/workspace-SR006.nfs3/layerwise-tucker-opt-20260917}
MODE=${1:-screen}
if [ "$MODE" = export ]; then
    python3 - "$RESULTS" "${2:-screen}" "${3:-}" <<'PY'
import base64
import gzip
import hashlib
import sys
from pathlib import Path
root = Path(sys.argv[1])
paths = sorted(root.glob('*correctness.json')) + sorted(root.glob('*micro.json')) if sys.argv[2] == 'checks' else sorted((root / sys.argv[2]).glob('*.json'))
for path in paths:
    if sys.argv[3] not in path.name:
        continue
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    encoded = base64.b64encode(gzip.compress(raw)).decode()
    chunks = [encoded[i:i+800] for i in range(0, len(encoded), 800)]
    for index, chunk in enumerate(chunks):
        print(f'RESULT_CHUNK {path.name} {index} {len(chunks)} {digest} {chunk}', flush=True)
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
for path in sorted(root.glob('*.exit')):
    print('APPLICATION_EXIT', path.stem, path.read_text().strip())
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
    python -m pip install --disable-pip-version-check --target /tmp/layerwise-opt-deps tiktoken loguru || return $?
    python -m pip install --disable-pip-version-check --no-deps --target /tmp/layerwise-opt-deps liger-kernel==0.8.1 || return $?
    python -m unittest discover -s tests -p 'test_layerwise*.py' -v || return $?
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
    if [ "$MODE" = riemann-pilot ]; then
        for fraction in 0.25 0.5; do
            timeout 1200 python scripts/validate_layerwise_riemannian.py --rank-fraction "$fraction" \
                --output "$RESULTS/$MODE-r$fraction-correctness.json" || return $?
        done
        for arm in dense:0.5 A:0.5 B:0.25; do
            variant=${arm%:*}; fraction=${arm#*:}
            optimizer=riemannian
            [ "$variant" = dense ] && optimizer=muon
            for policy in reference grouped grouped-qr; do
                [ "$variant" = dense ] && [ "$policy" = grouped-qr ] && continue
                implementation=reference; retraction=reference
                [ "$policy" != reference ] && implementation=grouped
                [ "$policy" = grouped-qr ] && retraction=grouped
                name="$variant-r$fraction-$policy"
                timeout 1200 python scripts/benchmark_layerwise_tucker.py --variant "$variant" \
                    --rank-fraction "$fraction" --optimizer "$optimizer" \
                    --optimizer-implementation "$implementation" --retraction-implementation "$retraction" \
                    --execution reordered --compile-mode max-autotune --microbatch 16 \
                    --warmup 5 --steps 10 --output "$RESULTS/$MODE/$name.json" || return $?
            done
        done
    elif [ "$MODE" = scaling ]; then
        width=2048; heads=16; ff=5632
        if [ "${2:-small}" = large ]; then
            width=2560; heads=20; ff=7040
        fi
        for fraction in 0.25 0.5; do
            for kernels in native liger; do
                extra=()
                [ "$kernels" = liger ] && extra=(--liger)
                timeout 900 python scripts/validate_layerwise_tucker_opt.py --compile-mode max-autotune \
                    --width "$width" --heads "$heads" --ffn-hidden-size "$ff" \
                    --layers 2 --accumulation 2 --stable-grad-buffers --rank-fraction "$fraction" \
                    "${extra[@]}" --output "$RESULTS/scaling-r$fraction-$kernels-correctness.json" || return $?
            done
        done
        python scripts/benchmark_layerwise_scaling.py --group "${2:-small}" --output-dir "$RESULTS/$MODE"
        return $?
    elif [ "$MODE" = audit-quarter ]; then
        execution=${2:-triton-pointwise}
        python scripts/validate_layerwise_tucker_opt.py --compile-mode max-autotune \
            --execution "$execution" --layers 12 --rank-fraction 0.25 \
            --output "$RESULTS/audit-quarter-$execution-correctness.json" || return $?
    elif [ "$MODE" = screen ]; then
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
    elif [ "$MODE" = compile ] || [ "$MODE" = compile-stable ]; then
        extra=()
        [ "$MODE" = compile-stable ] && extra=(--stable-grad-buffers)
        for compiler in reduce-overhead max-autotune; do
            python scripts/validate_layerwise_tucker_opt.py --compile-mode "$compiler" \
                --layers 12 --accumulation 2 "${extra[@]}" \
                --output "$RESULTS/$MODE-$compiler-correctness.json" || return $?
        done
        for mb in 1 16; do
            for arm in dense:0.5 A:0.5 B:0.25; do
                variant=${arm%:*}; fraction=${arm#*:}
                for compiler in reduce-overhead max-autotune; do
                    name="$variant-r$fraction-mb$mb-$compiler"
                    echo "BEGIN $name"
                    timeout 900 python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                        --microbatch "$mb" --execution reordered --compile-mode "$compiler" \
                        "${extra[@]}" \
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
    elif [ "$MODE" = combined ] || [ "$MODE" = combined-mb1 ]; then
        mb=16; accumulation=1; stable=(); executions=(reordered triton-pointwise)
        if [ "$MODE" = combined-mb1 ]; then
            mb=1; accumulation=2; stable=(--stable-grad-buffers); executions=(reordered)
        fi
        for execution in "${executions[@]}"; do
            python scripts/validate_layerwise_tucker_opt.py --compile-mode max-autotune --liger \
                --execution "$execution" --layers 12 --accumulation "$accumulation" "${stable[@]}" \
                --output "$RESULTS/$MODE-$execution-correctness.json" || return $?
        done
        for arm in dense:0.5 A:0.5 B:0.25; do
            variant=${arm%:*}; fraction=${arm#*:}
            for kernels in native liger liger-pointwise; do
                [ "$variant" = dense ] && [ "$kernels" = liger-pointwise ] && continue
                [ "$mb" = 1 ] && [ "$kernels" = liger-pointwise ] && continue
                extra=(); execution=reordered
                [ "$kernels" != native ] && extra=(--liger)
                [ "$kernels" = liger-pointwise ] && execution=triton-pointwise
                name="$variant-r$fraction-mb$mb-$kernels"
                echo "BEGIN $name"
                timeout 900 python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                    --microbatch "$mb" --execution "$execution" --compile-mode max-autotune "${extra[@]}" "${stable[@]}" \
                    --output "$RESULTS/$MODE/$name.json" || status=1
                echo "END $name"
            done
        done
    elif [ "$MODE" = confirm-mb16 ] || [ "$MODE" = confirm-mb1 ]; then
        kernels=${2:-native}
        if [ "$kernels" != native ] && [ "$kernels" != liger ]; then
            echo "Expected native or liger confirmation kernels"
            return 2
        fi
        mb=16; accumulation=1; stable=()
        arms=(dense:0.5 A:0.25 B:0.25 A:0.5 B:0.5)
        if [ "$MODE" = confirm-mb1 ]; then
            mb=1; accumulation=2; stable=(--stable-grad-buffers)
            arms=(dense:0.5 A:0.5 B:0.25)
        fi
        extra=()
        [ "$kernels" = liger ] && extra=(--liger)
        executions=(reordered triton-pointwise)
        [ "$mb" = 1 ] && executions=(reordered)
        for fraction in 0.25 0.5; do
            for execution in "${executions[@]}"; do
                timeout 900 python scripts/validate_layerwise_tucker_opt.py --compile-mode max-autotune \
                    --execution "$execution" --layers 12 --accumulation "$accumulation" --rank-fraction "$fraction" \
                    "${stable[@]}" "${extra[@]}" --output "$RESULTS/$MODE-r$fraction-$execution-correctness.json" || return $?
            done
        done
        for seed in 42 43 44; do
            policies=(reference optimized)
            [ "$mb" = 16 ] && policies+=(pointwise)
            if [ "$seed" = 43 ]; then
                policies=(optimized reference)
                [ "$mb" = 16 ] && policies=(pointwise optimized reference)
            fi
            for arm in "${arms[@]}"; do
                variant=${arm%:*}; fraction=${arm#*:}
                for policy in "${policies[@]}"; do
                    [ "$variant" = dense ] && [ "$policy" = pointwise ] && continue
                    options=(--execution reference)
                    if [ "$policy" != reference ]; then
                        options=(--execution reordered --compile-mode max-autotune "${stable[@]}" "${extra[@]}")
                        [ "$policy" = pointwise ] && options+=(--execution triton-pointwise)
                    fi
                    name="$variant-r$fraction-mb$mb-s$seed-$policy"
                    echo "BEGIN $name"
                    timeout 900 python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                        --microbatch "$mb" --seed "$seed" "${options[@]}" \
                        --output "$RESULTS/$MODE/$name.json" || return $?
                    echo "END $name"
                done
            done
        done
    elif [ "$MODE" = triton ]; then
        python scripts/benchmark_layerwise_triton.py --output "$RESULTS/triton-micro.json" || return $?
        for execution in triton-pointwise triton; do
            python scripts/validate_layerwise_tucker_opt.py --execution "$execution" \
                --output "$RESULTS/$execution-correctness.json" || return $?
        done
        for mb in 1 16; do
            for arm in A:0.5 B:0.25; do
                variant=${arm%:*}; fraction=${arm#*:}
                for execution in reordered triton-pointwise triton; do
                    name="$variant-r$fraction-mb$mb-$execution"
                    echo "BEGIN $name"
                    python scripts/benchmark_layerwise_tucker.py --variant "$variant" --rank-fraction "$fraction" \
                        --microbatch "$mb" --execution "$execution" \
                        --output "$RESULTS/$MODE/$name.json" || status=1
                    echo "END $name"
                done
            done
        done
    else
        echo "Unknown mode: $MODE"
        return 2
    fi
    return "$status"
}
run "$@" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
echo "APPLICATION_EXIT=$status"
echo "LOG=$LOG"
printf '%s\n' "$status" > "$RESULTS/$MODE.exit"
exit 0
