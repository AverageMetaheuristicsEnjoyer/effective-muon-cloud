#!/usr/bin/env bash
set -euo pipefail

OPT=${1:-numuon}
NGPUS=${2:-${NGPUS:-8}}
case "${OPT}" in
    adamw|muon|numuon) ;;
    *) echo "Usage: $0 {adamw|muon|numuon} [ngpus]" >&2; exit 2 ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd)
cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"

EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps_numuon_paper"}
WANDB_PROJECT=${WANDB_PROJECT:-"numuon-reproduction"}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}
WANDB=${WANDB:-0}

N_LAYER=${N_LAYER:-18}
N_EMBD=${N_EMBD:-1280}
N_HEAD=${N_HEAD:-20}
SEQ_LEN=${SEQ_LEN:-1024}
MULTIPLE_OF=${MULTIPLE_OF:-256}

ITERATIONS=${ITERATIONS:-2000}
WARMUP=${WARMUP:-500}
WSD_FRACT_DECAY=${WSD_FRACT_DECAY:-0.1}
WSD_FINAL_LR_SCALE=${WSD_FINAL_LR_SCALE:-0.0}
DECAY_TYPE=${DECAY_TYPE:-cosine}
BATCH_SIZE=${BATCH_SIZE:-16}
ACC_STEPS=${ACC_STEPS:-8}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
EVAL_BATCHES=${EVAL_BATCHES:-8}
LOG_INTERVAL=${LOG_INTERVAL:-50}
STABLE_RANK_LOG_INTERVAL=${STABLE_RANK_LOG_INTERVAL:-100}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-0}

case "${OPT}" in
    adamw)
        LR=${LR:-1e-3}
        WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
        SCHEDULER=${SCHEDULER:-cos}
        ;;
    muon)
        LR=${LR:-1e-3}
        WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
        SCHEDULER=${SCHEDULER:-wsd}
        ;;
    numuon)
        LR=${LR:-1e-3}
        WEIGHT_DECAY=${WEIGHT_DECAY:-1e-4}
        SCHEDULER=${SCHEDULER:-wsd}
        ;;
esac

EXP_NAME=${EXPERIMENT_NAME:-"${OPT}_0.5b_${ITERATIONS}steps"}
WANDB_GROUP=${WANDB_GROUP:-"numuon_0.5b_${ITERATIONS}steps"}

cmd=(
    torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py
    --distributed-backend nccl
    --experiment-name "${EXP_NAME}"
    --dataset fineweb
    --eval-cache-dir "${EVAL_CACHE_DIR}"
    --sequence-length "${SEQ_LEN}"
    --datasets-dir "${DATASETS_DIR}"
    --workers 8
    --model llama
    --n-layer "${N_LAYER}"
    --n-embd "${N_EMBD}"
    --n-head "${N_HEAD}"
    --multiple-of "${MULTIPLE_OF}"
    --dtype bfloat16
    --opt "${OPT}"
    --lr "${LR}"
    --weight-decay "${WEIGHT_DECAY}"
    --beta1 0.9
    --beta2 0.99
    --grad-clip 1.0
    --scheduler "${SCHEDULER}"
    --warmup-steps "${WARMUP}"
    --iterations "${ITERATIONS}"
    --wsd-fract-decay "${WSD_FRACT_DECAY}"
    --wsd-final-lr-scale "${WSD_FINAL_LR_SCALE}"
    --decay-type "${DECAY_TYPE}"
    --batch-size "${BATCH_SIZE}"
    --acc-steps "${ACC_STEPS}"
    --eval-interval "${EVAL_INTERVAL}"
    --eval-batches "${EVAL_BATCHES}"
    --log-interval "${LOG_INTERVAL}"
    --stable-rank-log-interval "${STABLE_RANK_LOG_INTERVAL}"
    --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}"
    --results-base-folder "${RESULTS_DIR}"
    --wandb-group "${WANDB_GROUP}"
)

if [[ "${OPT}" == "numuon" ]]; then
    cmd+=(
        --numuon-rank-start "${NUMUON_RANK_START:-1.0}"
        --numuon-rank-end "${NUMUON_RANK_END:-0.25}"
        --numuon-rank-scheduler "${NUMUON_RANK_SCHEDULER:-cosine}"
        --numuon-rank-warmup-fraction "${NUMUON_RANK_WARMUP_FRACTION:-0.1}"
        --numuon-rank-decay-end-fraction "${NUMUON_RANK_DECAY_END_FRACTION:-0.9}"
        --numuon-krylov-iters "${NUMUON_KRYLOV_ITERS:-2}"
        --numuon-oversample "${NUMUON_OVERSAMPLE:-8}"
    )
    if [[ "${NUMUON_WARM_START:-1}" == "0" ]]; then
        cmd+=(--numuon-no-warm-start)
    fi
fi

if [[ "${WANDB}" == "1" ]]; then
    cmd+=(--wandb --wandb-project "${WANDB_PROJECT}" --wandb-tags baseline bf16 0.5B "${OPT}" stable-rank)
fi

printf 'Running command:'
printf ' %q' "${cmd[@]}"
printf '\n'
"${cmd[@]}"
