#!/usr/bin/env bash
set -euo pipefail

ROOT=${TUCKER_LATE_ROOT:-/workspace-SR006.nfs3/tucker-late-growth-20260827}
PYTHON_DEPS="${ROOT}/python"
mkdir -p "${ROOT}/logs" "${ROOT}/exps" "${ROOT}/evals_cache" "${ROOT}/hf-cache" "${ROOT}/wandb" "${PYTHON_DEPS}"
export PYTHONNOUSERSITE=1
export PYTHONPATH="${PYTHON_DEPS}:.:src"
export HF_HOME="${ROOT}/hf-cache"

MODE=${1:-preflight}
echo "commit: $(git rev-parse HEAD)"
echo "mode: ${MODE}"
echo "mpi: ${OMPI_COMM_WORLD_RANK:-0}/${OMPI_COMM_WORLD_SIZE:-1}"
df -h /workspace-SR006.nfs3
df -i /workspace-SR006.nfs3

if [[ "${OMPI_COMM_WORLD_SIZE:-1}" != "1" ]]; then
    echo "Progressive Tucker requires exactly one Cloud MPI rank" >&2
    exit 2
fi

if [[ "${MODE}" == "repair-python" ]]; then
    rm -rf "${PYTHON_DEPS}"
    mkdir -p "${PYTHON_DEPS}"
    exit 0
fi

if [[ "${MODE}" == "peek" ]]; then
    newest=$(ls -t "${ROOT}"/logs/*.log 2>/dev/null | head -1)
    echo "latest log: ${newest:-none}"
    [[ -n "${newest}" ]] && tail -"${2:-200}" "${newest}"
    exit 0
fi

if [[ "${MODE}" == "disk" ]]; then
    du -h --max-depth=4 "${ROOT}" | sort -h | tail -40
    exit 0
fi

if [[ "${MODE}" == "preflight" ]]; then
    python - <<'PY'
import importlib
import os
import torch

print("python imports:")
for name in ("datasets", "huggingface_hub", "liger_kernel", "loguru", "pyarrow", "schedulefree", "tiktoken", "transformers", "wandb", "zstandard"):
    try:
        module = importlib.import_module(name)
        print(f"  {name}=ok version={getattr(module, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"  {name}=missing error={type(exc).__name__}: {exc}")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print(f"wandb_api_key_present={bool(os.environ.get('WANDB_API_KEY'))}")
PY
fi

if ! python -c "import loguru, schedulefree, sentry_sdk, tiktoken, wandb" 2>/dev/null; then
    pip install --target "${PYTHON_DEPS}" -q --no-deps \
        loguru==0.7.3 schedulefree sentry-sdk tiktoken==0.12.0 wandb==0.25.1
fi
python -c "import datasets, huggingface_hub, loguru, pyarrow, schedulefree, sentry_sdk, tiktoken, transformers, wandb, zstandard"
if [[ "${MODE}" == "preflight" ]]; then
    WANDB_MODE=offline WANDB_DIR="${ROOT}/wandb" python - <<'PY'
import wandb

run = wandb.init(project="tucker-cloud-preflight")
run.finish()
PY
    exit 0
fi

if [[ "${MODE}" == "correctness" ]]; then
    python -m unittest tests.test_progressive_tucker tests.test_tucker_chunked
    TUCKER_CUSTOM_CACHE_POLICY=recast \
        python experiments/fused_persistent_tucker/custom_backward/test_correctness.py
    exit 0
fi

case "${MODE}" in
    225) LAUNCHER=scripts/launch_tucker225_to_257_late_custom_backward.sh ;;
    169) LAUNCHER=scripts/launch_tucker169_to_257_late_custom_backward.sh ;;
    smoke225)
        LAUNCHER=scripts/launch_tucker225_to_257_late_custom_backward.sh
        export EXPERIMENT_NAME=llama257m_tucker_late_225m_to_257m_customfb_bs16acc8_smoke
        export ITERATIONS=2 WARMUP=1 EVAL_BATCHES=1 LATEST_CKPT_INTERVAL=2
        export DOWNSTREAM_EVAL_ENABLED=0 LM_EVAL_ENABLED=0 WANDB_MODE=disabled
        ;;
    *) echo "Expected mode: preflight, repair-python, peek, disk, correctness, smoke225, 225, or 169" >&2; exit 2 ;;
esac

export RESULTS_DIR="${ROOT}/exps"
export EVAL_CACHE_DIR="${ROOT}/evals_cache"
export DATASETS_DIR="${ROOT}/fineweb-local-if-present"
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
export WANDB_ENTITY=${WANDB_ENTITY:-efficient-muon}
export WANDB_PROJECT=${WANDB_PROJECT:-muon-variations}
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR=${WANDB_DIR:-"${ROOT}/wandb"}

LOG="${ROOT}/logs/${MODE}-$(date +%F_%H%M%S)-$$.log"
bash "${LAUNCHER}" 2>&1 | tee "${LOG}"
exit "${PIPESTATUS[0]}"
