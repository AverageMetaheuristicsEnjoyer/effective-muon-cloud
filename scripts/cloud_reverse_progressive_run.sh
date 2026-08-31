#!/usr/bin/env bash
set -euo pipefail

ROOT=${TUCKER_REVERSE_ROOT:-/workspace-SR006.nfs3/tucker-reverse-progressive-20260831}
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
    echo "Reverse Progressive Tucker requires exactly one Cloud MPI rank" >&2
    exit 2
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

if ! python -c "import loguru, schedulefree, sentry_sdk, tiktoken, wandb" 2>/dev/null; then
    pip install --target "${PYTHON_DEPS}" -q --no-deps \
        loguru==0.7.3 schedulefree sentry-sdk tiktoken==0.12.0 wandb==0.25.1
fi
if ! python -c "import importlib.metadata; assert importlib.metadata.version('ai2-olmo-eval') == '0.8.5'; from olmo_eval import HFTokenizer, ICLMetric, build_task" 2>/dev/null; then
    pip install --target "${PYTHON_DEPS}" -q --no-deps --upgrade \
        ai2-olmo-eval==0.8.5 torchmetrics==1.8.2 lightning-utilities==0.15.2 \
        cached-path==1.8.10 rich==13.9.4 importlib-resources==6.5.2
fi
python - "${PYTHON_DEPS}" <<'PY'
import importlib.metadata
import sys
from pathlib import Path

import torch
from olmo_eval import HFTokenizer, ICLMetric, build_task

python_deps = Path(sys.argv[1]).resolve()
torch_path = Path(torch.__file__).resolve()
if torch.__version__.split("+", 1)[0] != "2.8.0":
    raise RuntimeError(f"Expected system torch 2.8.0, found {torch.__version__} at {torch_path}")
if torch_path.is_relative_to(python_deps):
    raise RuntimeError(f"Refusing shadow torch installation at {torch_path}")
if importlib.metadata.version("ai2-olmo-eval") != "0.8.5":
    raise RuntimeError("Expected ai2-olmo-eval==0.8.5")
print(f"torch={torch.__version__} cuda={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
print("ai2-olmo-eval=0.8.5 import=ok")
PY
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
    python - <<'PY'
from tests.test_progressive_tucker import (
    test_rank_shrink_projects_weight_and_optimizer_state,
    test_reverse_progressive_controller_uses_requested_final_ranks,
)

test_rank_shrink_projects_weight_and_optimizer_state()
test_reverse_progressive_controller_uses_requested_final_ranks()
print("reverse progressive CPU tests: ok")
PY
    TUCKER_CUSTOM_CACHE_POLICY=recast \
        python experiments/fused_persistent_tucker/custom_backward/test_correctness.py
    exit 0
fi

LAUNCHER=scripts/launch_tucker257_to_133_reverse_progressive_custom_backward.sh
if [[ "${MODE}" == "smoke" ]]; then
    export EXPERIMENT_NAME=llama257m_tucker_reverse_progressive_customfb_smoke
    export TUCKER_PROGRESSIVE_STAGES="0:257676352 1:225000000 2:190000000 3:160000000 4:133000000"
    export ITERATIONS=5 WARMUP=1 BATCH_SIZE=1 ACC_STEPS=1 EVAL_BATCHES=1 LATEST_CKPT_INTERVAL=5
    export DOWNSTREAM_EVAL_ENABLED=0 LM_EVAL_ENABLED=0 WANDB_MODE=disabled
elif [[ "${MODE}" != "train" ]]; then
    echo "Expected mode: preflight, correctness, smoke, train, peek, or disk" >&2
    exit 2
fi

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
