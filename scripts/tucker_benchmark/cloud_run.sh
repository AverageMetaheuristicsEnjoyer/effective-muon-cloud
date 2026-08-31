#!/bin/bash
# Cloud.ru mlsub entry point. It keeps logs and results on the persistent nfs3 volume.
set -u

RESULTS=${TUCKER_BENCH_RESULTS:-/workspace-SR006.nfs3/tucker-order3-screen-20260901}
mkdir -p "$RESULTS/logs" "$RESULTS/results"

if [ "${1:-}" = "disk" ]; then
    df -h /workspace-SR006.nfs3
    df -i /workspace-SR006.nfs3
    du -sh /workspace-SR006.nfs3/tucker-membench/python 2>/dev/null || true
    du -sh "$RESULTS" 2>/dev/null || true
    exit 0
fi

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

print("model\tvariant\trank_profile\tmode_layout\tparameters\tdifference_from_dense\tmicrobatch\tstatus\tmedian_ms\tforward_ms\tbackward_ms\tforward_backward_ms\tforward_backward_tokens_per_second\toptimizer_ms\tretraction_ms\tforward_backward_peak_gb\tfull_peak_gb\treserved_gb\tstate_gb\tmodel_gb\tgrad_clip_ms\tstate_dtypes\tgpu")
for path in sorted(Path(sys.argv[1]).glob("*.json")):
    payload = json.loads(path.read_text())
    status = payload.get("status")
    if status != "complete":
        controls = payload.get("requested_controls", {})
        print(f"{controls.get('model_size', '')}\t{payload.get('variant', '')}\t{controls.get('tucker_rank_profile', 'iso')}\t{controls.get('tucker_mode_layout', 'balanced4')}\t\t\t{controls.get('microbatch', '')}\t{status}")
        continue
    summary = payload["summary"]
    memory = payload["memory"]
    print("\t".join(map(str, (
        payload["model"]["name"],
        payload["variant"]["name"],
        payload["benchmark"].get("tucker_rank_profile", "iso"),
        payload["benchmark"].get("tucker_mode_layout", "balanced4"),
        payload["model"]["actual_parameters"],
        payload["model"]["parameter_difference_from_dense"],
        payload["benchmark"]["microbatch"],
        status,
        round(summary["host_total_ms"]["median"], 3),
        round(summary["forward_ms"]["median"], 3),
        round(summary["backward_ms"]["median"], 3),
        round(summary["forward_backward_ms"]["median"], 3),
        round(summary["tokens_per_second_forward_backward"]["median"], 1),
        round(summary["optimizer_ms"]["median"], 3),
        round(summary["retraction_ms"]["median"], 3),
        round(memory["forward_backward_peak_allocated_bytes"] / 1e9, 3),
        round(memory["peak_allocated_bytes"] / 1e9, 3),
        round(memory["peak_reserved_bytes"] / 1e9, 3),
        round(memory["optimizer_state_bytes"] / 1e9, 3),
        round(memory["model_bytes"] / 1e9, 3),
        round(summary["grad_clip_ms"]["median"], 3),
        ",".join(memory["optimizer_state_dtypes"]),
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
    PYTHON_DEPS=/tmp/tucker-benchmark-python
    export PYTHONPATH="$PYTHON_DEPS:$PYTHONPATH"
    if ! python -c "import tiktoken, loguru, liger_kernel" 2>/dev/null; then
        mkdir -p "$PYTHON_DEPS"
        pip install --target "$PYTHON_DEPS" -q --no-deps \
            tiktoken loguru liger-kernel==0.8.1
    fi

    run_correctness() {
        python experiments/fused_persistent_tucker/custom_backward/test_correctness.py && \
            python experiments/fused_persistent_tucker/custom_backward/test_parallel_muon.py && \
            python experiments/fused_persistent_tucker/custom_backward/test_grouped_retraction.py
    }

    run_autotune() {
        python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/autotune-streams-1" \
            --models 257m --variants tucker_parallel --microbatches 16 \
            --warmup-steps 3 --measured-steps 12 --tucker-muon-streams 1 && \
        python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/autotune-streams-2" \
            --models 257m --variants tucker_parallel --microbatches 16 \
            --warmup-steps 3 --measured-steps 12 --tucker-muon-streams 2 && \
        python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/autotune-streams-4" \
            --models 257m --variants tucker_parallel --microbatches 16 \
            --warmup-steps 3 --measured-steps 12 --tucker-muon-streams 4
    }

    if [ "${1:-}" = "selftest" ]; then
        python -m unittest discover -s tests -p test_tucker_benchmark.py
    elif [ "${1:-}" = "correctness" ]; then
        run_correctness
    elif [ "${1:-}" = "autotune" ]; then
        run_autotune
    elif [ "${1:-}" = "progressive-ranks" ]; then
        run_correctness && python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/results" \
            --models 257m \
            --variants tucker_reference,tucker_parallel \
            --rank-profiles progressive_133m_exact,progressive_133m_rank8,progressive_160m_exact,progressive_160m_rank8,progressive_190m_exact,progressive_190m_rank8,progressive_225m_exact,progressive_225m_rank8 \
            --tucker-muon-streams 4
    elif [ "${1:-}" = "order3-screen" ]; then
        run_correctness && python -m scripts.tucker_benchmark.run_sweep \
            --output-dir "$RESULTS/results" \
            --models 257m \
            --variants tucker_parallel \
            --rank-profiles progressive_133m_rank8,progressive_225m_rank8 \
            --mode-layouts balanced4,order3_input,order3_output \
            --microbatches 1,16 \
            --tucker-muon-streams 4
    elif [ "${1:-}" = "pipeline" ] || [ "${1:-}" = "pipeline-257m" ]; then
        run_correctness && run_autotune
        pipeline_status=$?
        if [ "$pipeline_status" -eq 0 ]; then
            best_streams=$(python - "$RESULTS" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
scores = {}
for streams in (1, 2, 4):
    path = root / f"autotune-streams-{streams}/runs/257m-tucker_parallel-bs16.json"
    payload = json.loads(path.read_text())
    scores[streams] = payload["summary"]["optimizer_ms"]["median"]
print(min(scores, key=scores.get))
PY
            )
            echo "selected H100 Muon streams: $best_streams"
            model_args=()
            if [ "${1:-}" = "pipeline-257m" ]; then
                model_args=(--models 257m)
            fi
            python -m scripts.tucker_benchmark.run_sweep \
                --output-dir "$RESULTS/results" \
                --tucker-muon-streams "$best_streams" \
                "${model_args[@]}"
        else
            (exit "$pipeline_status")
        fi
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
