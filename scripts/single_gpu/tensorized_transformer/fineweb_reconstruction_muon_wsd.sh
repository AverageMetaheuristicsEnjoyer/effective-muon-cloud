#!/usr/bin/env bash
set -euo pipefail

# Tensorized reconstruction + pure all-Linear Tucker run for FineWeb 1x-Chinchilla.
# This mirrors muonbp_lr1e-3.sh except for:
#   - plain Muon instead of MuonBP;
#   - tensorized reconstruction attention instead of standard attention;
#   - every independent Linear (including lm_head) uses only Tucker factors/core.
# A single scalar Tucker rank 259 gives 257,181,058 parameters: 7,806 fewer
# than the original 257,188,864 control and within the allowed 12,312 gap.
# No sparse correction, residual, filler, or direct parameter matching is used.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NGPUS=${1:-1}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"tensorized_reconstruction_r1023_tucker_r259_pure_muon_lr1e-3_1xC"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256
TENSORIZED_RANK=${TENSORIZED_RANK:-1023}
TENSORIZED_NUM_CORES=2
LINEAR_PARAMETERIZATION=${LINEAR_PARAMETERIZATION:-"tucker"}
TARGET_PARAMETER_COUNT=${TARGET_PARAMETER_COUNT:-257188864}
TARGET_PARAMETER_TOLERANCE=${TARGET_PARAMETER_TOLERANCE:-12312}
TUCKER_RANK=${TUCKER_RANK:-259}
TUCKER_RANKS=${TUCKER_RANKS:-""} # optional explicit r1,r2,r3,r4
TUCKER_EQUAL_PARAMS=${TUCKER_EQUAL_PARAMS:-0}
TUCKER_FORWARD_MODE=${TUCKER_FORWARD_MODE:-"auto"}

ITERATIONS=39250          # 1x Chinchilla for 257M (= 157000 / 4)
WARMUP=2000
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
BATCH_SIZE=32
ACC_STEPS=4
LR=1e-3
WEIGHT_DECAY=0.1

WANDB_ENTITY_ARG=()
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARG=(--wandb-entity "${WANDB_ENTITY}")
fi

TUCKER_EQUAL_PARAMS_FLAG=--tucker-equal-params
TUCKER_MODE_TAG=equal-params
if [[ "${TUCKER_EQUAL_PARAMS}" == "0" ]]; then
    TUCKER_EQUAL_PARAMS_FLAG=--no-tucker-equal-params
    TUCKER_MODE_TAG=pure-tucker
fi

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NGPUS}" \
    src/main.py \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    \
    --dataset fineweb \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length ${SEQ_LEN} \
    --streaming \
    --datasets-dir "${DATASETS_DIR}" \
    --workers 8 \
    \
    --model llama \
    --linear-parameterization "${LINEAR_PARAMETERIZATION}" \
    --target-parameter-count "${TARGET_PARAMETER_COUNT}" \
    --target-parameter-tolerance "${TARGET_PARAMETER_TOLERANCE}" \
    --tucker-rank "${TUCKER_RANK}" \
    --tucker-ranks "${TUCKER_RANKS}" \
    --tucker-forward-mode "${TUCKER_FORWARD_MODE}" \
    "${TUCKER_EQUAL_PARAMS_FLAG}" \
    --attention-type tensorized \
    --tensorized-mode reconstruction \
    --tensorized-rank ${TENSORIZED_RANK} \
    --tensorized-num-cores ${TENSORIZED_NUM_CORES} \
    --tensorized-causal \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt muon \
    --lr ${LR} \
    --momentum 0.95 \
    --lite-muon-theta 0.95 \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 0.9 \
    --beta2 0.99 \
    --grad-clip 1.0 \
    \
    --scheduler wsd \
    --warmup-steps ${WARMUP} \
    --iterations ${ITERATIONS} \
    --wsd-fract-decay ${WSD_FRACT_DECAY} \
    --wsd-final-lr-scale ${WSD_FINAL_LR_SCALE} \
    --decay-type ${DECAY_TYPE} \
    \
    --batch-size ${BATCH_SIZE} \
    --acc-steps ${ACC_STEPS} \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --downstream-eval-enabled \
    --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 2000 \
    --lm-eval-datasets wikitext103 \
    --log-interval 50 \
    --stable-rank-interval 1000 \
    --stable-rank-log-spectrum \
    \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    ${WANDB_ENTITY_ARG[@]+"${WANDB_ENTITY_ARG[@]}"} \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group 1xChinchilla \
    --wandb-tags bf16 muon streaming tensorized reconstruction tucker all-linear "${TUCKER_MODE_TAG}"
