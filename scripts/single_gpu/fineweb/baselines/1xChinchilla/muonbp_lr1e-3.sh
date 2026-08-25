#!/usr/bin/env bash
set -euo pipefail

# MuonBP (block-periodic Muon, arXiv:2510.16981) — 257M, 1x Chinchilla, single GPU.
#   - period P = 5            (paper-recommended optimal)
#   - nblocks  = 2            (paper's small-model TP=2 sharding)
#   - --lr 1e-3               ("full" learning rate, as in the repo baselines)
#   - block-step learning rate follows automatically from the paper's
#     dimension-based RMS scaling (no separate number to set)
#   - stable-rank / spectrum metrics logged to W&B (same as the stable_rank repo)

NGPUS=${1:-1}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

ITERATIONS=39250          # 1x Chinchilla for 257M (= 157000 / 4)
WARMUP=2000
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
BATCH_SIZE=32
ACC_STEPS=4
LR=1e-3
WEIGHT_DECAY=0.1
MUONBP_PERIOD=5
MUONBP_NBLOCKS=2

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "muonbp_lr1e-3_1xC" \
    \
    --dataset fineweb \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length ${SEQ_LEN} \
    --streaming \
    --datasets-dir "${DATASETS_DIR}" \
    --workers 8 \
    \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd  ${N_EMBD} \
    --n-head  ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt muonbp \
    --muonbp-period ${MUONBP_PERIOD} \
    --muonbp-nblocks ${MUONBP_NBLOCKS} \
    --lr ${LR} \
    --momentum 0.95 \
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
    --stable-rank-interval 500 \
    --stable-rank-log-spectrum \
    \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    ${WANDB_ENTITY:+--wandb-entity "${WANDB_ENTITY}"} \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group 1xChinchilla \
    --wandb-tags baseline bf16 muonbp streaming
