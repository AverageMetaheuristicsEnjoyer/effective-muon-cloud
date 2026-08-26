#!/usr/bin/env bash
set -euo pipefail

NGPUS=${1:-2}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}

N_LAYER=18
N_EMBD=1280
N_HEAD=20
SEQ_LEN=1024
MULTIPLE_OF=256

ITERATIONS=75456
WARMUP=2000
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine
BATCH_SIZE=16
ACC_STEPS=8
LR=1e-4
WEIGHT_DECAY=1e-4

BETA1=0.9
BETA2=0.999
BETA3=0.9999
ALPHA=8
BETA3_WARMUP=${ITERATIONS}
ALPHA_WARMUP=${ITERATIONS}

torchrun --standalone --nproc_per_node="${NGPUS}" src/main.py \
    --distributed-backend nccl \
    --experiment-name "ademamix_lr1e-3_1xC_2gpus" \
    \
    --dataset fineweb \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length ${SEQ_LEN} \
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
    --opt ademamix \
    --lr ${LR} \
    --weight-decay ${WEIGHT_DECAY} \
    --beta1 ${BETA1} \
    --beta2 ${BETA2} \
    --ademamix_beta3 ${BETA3} \
    --ademamix_alpha ${ALPHA} \
    --ademamix_beta3_warmup_steps ${BETA3_WARMUP} \
    --ademamix_alpha_warmup_steps ${ALPHA_WARMUP} \
    --grad-clip 0.5 \
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
    \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group 1xChinchilla_2gpus \
    --wandb-tags baseline bf16 0.5B ademamix
