#!/usr/bin/env bash
set -euo pipefail

# Tensorized Transformer / Multi-linear Attention (arXiv:1906.09777).
#
# Defaults use the parameter-matched reconstruction setup selected for this
# project: d_model=256, d_ff=2100, 6 layers, rank=314, core-2, sequence length
# 80. Adam, inverse-sqrt schedule, 4k warmup, dropout=0.1, and label smoothing
# 0.1 follow the WikiText-103 setup from the paper. Set TENSORIZED_MODE to
# split_concat and TENSORIZED_RANK to 40 to run the paper's Eq. (8) setup.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NGPUS=${1:-1}
DATASETS_DIR=${DATASETS_DIR:-"./datasets"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"tensorized-transformer"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"tensorized_reconstruction_r314_wt103"}

N_LAYER=${N_LAYER:-6}
N_EMBD=${N_EMBD:-256}
N_HEAD=${N_HEAD:-1}
FFN_HIDDEN_SIZE=${FFN_HIDDEN_SIZE:-2100}
SEQ_LEN=${SEQ_LEN:-80}
TENSORIZED_RANK=${TENSORIZED_RANK:-314}
TENSORIZED_NUM_CORES=${TENSORIZED_NUM_CORES:-2}
TENSORIZED_MODE=${TENSORIZED_MODE:-"reconstruction"}
TENSORIZED_QUERY_CHUNK_SIZE=${TENSORIZED_QUERY_CHUNK_SIZE:-8}
TENSORIZED_CAUSAL=${TENSORIZED_CAUSAL:-"true"}

ITERATIONS=${ITERATIONS:-200000}
WARMUP_STEPS=${WARMUP_STEPS:-4000}
BATCH_SIZE=${BATCH_SIZE:-8}
ACC_STEPS=${ACC_STEPS:-8}
LR=${LR:-2.5e-4}

case "${TENSORIZED_CAUSAL}" in
    1|true|TRUE|yes|YES) CAUSAL_ARG=(--tensorized-causal) ;;
    0|false|FALSE|no|NO) CAUSAL_ARG=(--no-tensorized-causal) ;;
    *) echo "TENSORIZED_CAUSAL must be true or false" >&2; exit 2 ;;
esac

WANDB_ENTITY_ARG=()
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARG=(--wandb-entity "${WANDB_ENTITY}")
fi

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    \
    --dataset wikitext \
    --datasets-dir "${DATASETS_DIR}" \
    --sequence-length "${SEQ_LEN}" \
    \
    --model base \
    --attention-type tensorized \
    --tensorized-mode "${TENSORIZED_MODE}" \
    --tensorized-rank "${TENSORIZED_RANK}" \
    --tensorized-num-cores "${TENSORIZED_NUM_CORES}" \
    --tensorized-query-chunk-size "${TENSORIZED_QUERY_CHUNK_SIZE}" \
    "${CAUSAL_ARG[@]}" \
    --n-layer "${N_LAYER}" \
    --n-embd "${N_EMBD}" \
    --n-head "${N_HEAD}" \
    --ffn-hidden-size "${FFN_HIDDEN_SIZE}" \
    --dropout 0.1 \
    --label-smoothing 0.1 \
    --dtype bfloat16 \
    \
    --opt adamw \
    --lr "${LR}" \
    --beta1 0.9 \
    --beta2 0.999 \
    --weight-decay 0.0 \
    --grad-clip 0.25 \
    \
    --scheduler inverse_sqrt \
    --warmup-steps "${WARMUP_STEPS}" \
    --iterations "${ITERATIONS}" \
    --batch-size "${BATCH_SIZE}" \
    --acc-steps "${ACC_STEPS}" \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --log-interval 50 \
    --stable-rank-interval 500 \
    --stable-rank-log-spectrum \
    \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    "${WANDB_ENTITY_ARG[@]}" \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group tensorized-transformer-wikitext103 \
    --wandb-tags tensorized-transformer multi-linear-attention btd core-2 reconstruction rank-314 arxiv-1906.09777
