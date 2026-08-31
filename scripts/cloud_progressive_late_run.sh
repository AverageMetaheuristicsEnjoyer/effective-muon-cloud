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

if [[ "${MODE}" == "peek" || "${MODE}" == "peek-225" || "${MODE}" == "peek-169" ]]; then
    pattern="*.log"
    [[ "${MODE}" != "peek" ]] && pattern="${MODE#peek-}-*.log"
    newest=$(ls -t "${ROOT}"/logs/${pattern} 2>/dev/null | head -1)
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
print(f"torch={torch.__version__} path={torch_path}")
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

if [[ "${MODE}" == "resume-preflight" ]]; then
    python - "${ROOT}" <<'PY'
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
experiments = {
    "225": "llama257m_tucker_late_225m_to_257m_customfb_bs16acc8_run2",
    "169": "llama257m_tucker_late_169m_to_257m_customfb_bs16acc8_run1",
}
for arm, experiment in experiments.items():
    ckpt_root = root / "exps" / "1xChinchilla-tucker-retract" / experiment / "ckpts"
    for name in ("latest", "best_val"):
        ckpt_dir = ckpt_root / name
        sizes = {
            path.name: path.stat().st_size
            for path in (ckpt_dir / "main.pt", ckpt_dir / "worker_0.pt")
            if path.is_file()
        }
        print(f"arm={arm} checkpoint={name} sizes={sizes}")

    latest = ckpt_root / "latest"
    main = torch.load(latest / "main.pt", map_location="cpu", weights_only=False)
    worker = torch.load(latest / "worker_0.pt", map_location="cpu", weights_only=False)
    if int(main["itr"]) != 2000:
        raise RuntimeError(f"arm={arm}: expected iter 2000, found {main['itr']}")
    if "train_reader_state" not in worker:
        raise RuntimeError(f"arm={arm}: checkpoint has no FineWeb reader state")
    print(f"arm={arm} latest_iter={main['itr']} load=ok reader_state=ok")
    del main, worker
PY
    exit 0
fi

if [[ "${MODE}" == "archive-225" || "${MODE}" == "archive-169" ]]; then
    python - "${ROOT}" "${MODE#archive-}" <<'PY'
import os
import shutil
import sys
from pathlib import Path

import torch
from huggingface_hub import HfApi

root = Path(sys.argv[1])
arm = sys.argv[2]
experiments = {
    "225": "llama257m_tucker_late_225m_to_257m_customfb_bs16acc8_run2",
    "169": "llama257m_tucker_late_169m_to_257m_customfb_bs16acc8_run1",
}
repo_id = os.environ["HF_CHECKPOINT_REPO"]
token = os.environ["HF_TOKEN"]
ckpt_root = root / "exps" / "1xChinchilla-tucker-retract" / experiments[arm] / "ckpts"
latest = ckpt_root / "latest"
main_path = latest / "main.pt"
worker_path = latest / "worker_0.pt"
before = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in (main_path, worker_path)}
main = torch.load(main_path, map_location="cpu", weights_only=False)
worker = torch.load(worker_path, map_location="cpu", weights_only=False)
iteration = int(main["itr"])
if "train_reader_state" not in worker:
    raise RuntimeError("Checkpoint has no FineWeb reader state")
del main, worker

path_in_repo = f"{arm}/iter_{iteration:08d}"
api = HfApi(token=token)
commit = api.upload_folder(
    folder_path=latest,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Archive {arm} arm checkpoint at iter {iteration}",
)
repo_files = set(api.list_repo_files(repo_id=repo_id, repo_type="model"))
expected = {f"{path_in_repo}/main.pt", f"{path_in_repo}/worker_0.pt"}
if not expected.issubset(repo_files):
    raise RuntimeError(f"HF upload verification failed: missing {sorted(expected - repo_files)}")
after = {path.name: (path.stat().st_size, path.stat().st_mtime_ns) for path in (main_path, worker_path)}
if after != before:
    raise RuntimeError("Checkpoint changed during upload; keeping local files")

shutil.rmtree(latest)
best_val = ckpt_root / "best_val"
best_main = best_val / "main.pt"
if best_main.is_file():
    best = torch.load(best_main, map_location="meta", weights_only=False)
    best_iteration = int(best["itr"])
    del best
    if best_iteration == iteration:
        shutil.rmtree(best_val)
print(f"archived arm={arm} iter={iteration} commit={commit.oid} local_deleted=true")
PY
    exit 0
fi

if [[ "${MODE}" == "correctness" ]]; then
    python -m unittest \
        tests.test_progressive_tucker \
        tests.test_tucker_chunked \
        tests.test_tucker_linear \
        tests.test_tucker_benchmark
    TUCKER_CUSTOM_CACHE_POLICY=recast \
        python experiments/fused_persistent_tucker/custom_backward/test_correctness.py
    exit 0
fi

if [[ "${MODE}" == "diagnose-169-growth" ]]; then
    CKPT="${ROOT}/exps/1xChinchilla-tucker-retract/llama257m_tucker_late_169m_to_257m_customfb_bs16acc8_run1/ckpts/latest"
    DIAGNOSTIC_LOG="${ROOT}/logs/diagnose-169-growth-$(date +%F_%H%M%S)-$$.log"
    python scripts/diagnose_progressive_growth.py --checkpoint "${CKPT}" 2>&1 | tee "${DIAGNOSTIC_LOG}"
    exit "${PIPESTATUS[0]}"
fi

if [[ "${MODE}" == "inspect-225-checkpoint" || "${MODE}" == "inspect-169-checkpoint" ]]; then
    ARM="${MODE#inspect-}"
    ARM="${ARM%-checkpoint}"
    python - "${ROOT}" "${ARM}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
arm = sys.argv[2]
experiments = {
    "225": "llama257m_tucker_late_225m_to_257m_customfb_bs16acc8_run2",
    "169": "llama257m_tucker_late_169m_to_257m_customfb_bs16acc8_run1",
}
latest = root / "exps" / "1xChinchilla-tucker-retract" / experiments[arm] / "ckpts" / "latest"
main_path = latest / "main.pt"
worker_path = latest / "worker_0.pt"
before = {
    path.name: (path.stat().st_size, path.stat().st_mtime_ns)
    for path in (main_path, worker_path)
}
main = torch.load(main_path, map_location="cpu", weights_only=False)
worker = torch.load(worker_path, map_location="cpu", weights_only=False)
after = {
    path.name: (path.stat().st_size, path.stat().st_mtime_ns)
    for path in (main_path, worker_path)
}
if after != before:
    raise RuntimeError("Checkpoint changed during inspection")
if "train_reader_state" not in worker:
    raise RuntimeError("Checkpoint has no FineWeb reader state")
print(
    json.dumps(
        {
            "arm": arm,
            "iteration": int(main["itr"]),
            "main_bytes": before["main.pt"][0],
            "worker_bytes": before["worker_0.pt"][0],
            "optimizer_state_entries": len(main["optimizer"]["state"]),
            "progressive_stage": int(main["progressive_tucker"]["stage_index"]),
            "reader_state": True,
            "stable_during_load": True,
        },
        sort_keys=True,
    )
)
PY
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
    order3|smoke-order3)
        TUCKER_MODE_LAYOUT=${2:?order3 mode requires order3_input or order3_output}
        case "${TUCKER_MODE_LAYOUT}" in
            order3_input|order3_output) ;;
            *) echo "Expected order3_input or order3_output" >&2; exit 2 ;;
        esac
        TUCKER_RANK_PLAN="${ROOT}/plans/${TUCKER_MODE_LAYOUT}-225m-rank8.json"
        python scripts/make_tucker_order3_rank_plan.py \
            --layout "${TUCKER_MODE_LAYOUT}" \
            --profile progressive_225m_rank8 \
            --output "${TUCKER_RANK_PLAN}"
        export TUCKER_MODE_LAYOUT TUCKER_RANK_PLAN
        LAUNCHER=scripts/launch_tucker3_225_to_257_late_custom_backward.sh
        if [[ "${MODE}" == "smoke-order3" ]]; then
            export EXPERIMENT_NAME="llama257m_tucker3_${TUCKER_MODE_LAYOUT}_late_225m_to_257m_customfb_bs16acc8_smoke"
            export ITERATIONS=2 WARMUP=1 EVAL_BATCHES=1 LATEST_CKPT_INTERVAL=2
            export DOWNSTREAM_EVAL_ENABLED=0 LM_EVAL_ENABLED=0 WANDB_MODE=disabled
        fi
        ;;
    *) echo "Expected a documented preflight, inspection, smoke, or training mode" >&2; exit 2 ;;
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
