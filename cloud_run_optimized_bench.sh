#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-all}"
ARTIFACT_ROOT="${BENCH_ARTIFACT_ROOT:-/workspace-SR006.nfs3/progressive-tucker-257m-optimized-20260825}"
RUN_STAMP="$(date +%F_%H%M%S)-$$"

mkdir -p \
    "$ARTIFACT_ROOT/logs" \
    "$ARTIFACT_ROOT/evals_cache" \
    "$ARTIFACT_ROOT/exps" \
    "$ARTIFACT_ROOT/hf_cache" \
    "$ARTIFACT_ROOT/pip_cache" \
    "$ARTIFACT_ROOT/python_user" \
    "$ARTIFACT_ROOT/wandb"

LOG_PATH="$ARTIFACT_ROOT/logs/${RUN_STAMP}-${MODE}.log"
exec > >(tee -a "$LOG_PATH") 2>&1

export PYTHONUNBUFFERED=1
export PYTHONUSERBASE="$ARTIFACT_ROOT/python_user"
export PATH="$PYTHONUSERBASE/bin:$PATH"
export PIP_USER=1
export PIP_CACHE_DIR="$ARTIFACT_ROOT/pip_cache"
export HF_HOME="$ARTIFACT_ROOT/hf_cache"
export WANDB_MODE=disabled
export WANDB_DIR="$ARTIFACT_ROOT/wandb"
export EVAL_CACHE_DIR="$ARTIFACT_ROOT/evals_cache"
export RESULTS_DIR="$ARTIFACT_ROOT/exps"
export MASTER_PORT="${MASTER_PORT:-29725}"

echo "mode: $MODE"
echo "started: $(date -Is)"
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
echo "log: $LOG_PATH"
python --version
nvcc --version || true
nvidia-smi || true
df -h "$ARTIFACT_ROOT" || true
df -i "$ARTIFACT_ROOT" || true

bash setup.sh
setup_status=$?
echo "SETUP_EXIT=$setup_status"
if [ "$setup_status" -ne 0 ]; then
    exit 0
fi

python - <<'PY'
import flash_attn
import torch
import torchvision

print(f"torch: {torch.__version__}")
print(f"torchvision: {torchvision.__version__}")
print(f"flash_attn: {flash_attn.__version__}")
print(f"cuda_available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu: {torch.cuda.get_device_name(0)}")
    print(f"gpu_memory_bytes: {torch.cuda.get_device_properties(0).total_memory}")
PY

if [ "$MODE" = "setup" ]; then
    echo "FINAL_EXIT=0"
    exit 0
fi

time_status=0
memory_status=0

if [ "$MODE" = "all" ] || [ "$MODE" = "time" ]; then
    echo "=== TIME BENCH START $(date -Is) ==="
    TIME_BENCH=1 bash run_optimized_tucker.sh
    time_status=$?
    echo "TIME_BENCH_EXIT=$time_status"
    echo "=== TIME BENCH END $(date -Is) ==="
fi

if [ "$MODE" = "all" ] || [ "$MODE" = "memory" ]; then
    echo "=== MEMORY BENCH START $(date -Is) ==="
    MEMORY_BENCH=1 bash run_optimized_tucker.sh
    memory_status=$?
    echo "MEMORY_BENCH_EXIT=$memory_status"
    echo "=== MEMORY BENCH END $(date -Is) ==="
fi

if [ "$time_status" -ne 0 ] || [ "$memory_status" -ne 0 ]; then
    echo "FINAL_EXIT=1"
else
    echo "FINAL_EXIT=0"
fi
exit 0
