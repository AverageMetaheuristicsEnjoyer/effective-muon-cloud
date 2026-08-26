#!/usr/bin/env bash
set -euo pipefail

# One-GPU 257M Llama, trained for exactly 1x Chinchilla on FineWeb-edu.
# The 84 internal Q/K/V/O/Gate/Up/Down matrices use rank-only Tucker maps.
# lm_head intentionally remains a dense nn.Linear.
# After each optimizer step, QR gauge fixing makes all Tucker factors
# column-orthonormal and absorbs R into the core without changing W_effective.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NGPUS=1
PYTHON_BIN=${PYTHON_BIN:-python}
MAIN_SCRIPT=${MAIN_SCRIPT:-src/main.py}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"tucker-experiments"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"llama257m_tucker_r259_muon_retract_1x_chinchilla"}
OPTIMIZED_KERNELS=${OPTIMIZED_KERNELS:-1}
TORCH_COMPILE=${TORCH_COMPILE:-0}
ACTIVATION_CHECKPOINTING=${ACTIVATION_CHECKPOINTING:-0}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256

TARGET_PARAMETER_COUNT=257676352
TARGET_PARAMETER_TOLERANCE=12312
TUCKER_RANK=259
TUCKER_FORWARD_MODE=${TUCKER_FORWARD_MODE:-chunked_contract}
# Tuned on A100 PCIe for the production 16x1024 microbatch.
TUCKER_CONTRACT_CHUNK_SIZE=${TUCKER_CONTRACT_CHUNK_SIZE:-16384}
TUCKER_HEAD_CONTRACT_CHUNK_SIZE=${TUCKER_HEAD_CONTRACT_CHUNK_SIZE:-2048}

# 39,250 * 16 * 8 * 1,024 = 5,144,576,000 tokens, approximately
# 19.97 tokens per parameter for the 257,676,352-parameter Tucker model.
ITERATIONS=39250
BATCH_SIZE=16
ACC_STEPS=8
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}

LR=1e-3
MUON_MOMENTUM=0.95
ADAMW_BETA1=0.9
ADAMW_BETA2=0.99
WEIGHT_DECAY=0.1
GRAD_CLIP=1.0

WARMUP=2000
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine

WANDB_ENTITY_ARG=()
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARG=(--wandb-entity "${WANDB_ENTITY}")
fi

CHECKPOINTING_ARG=()
if [[ "${ACTIVATION_CHECKPOINTING}" == "1" ]]; then
    CHECKPOINTING_ARG=(--activation-checkpointing)
fi

OPTIMIZED_KERNEL_ARGS=()
if [[ "${OPTIMIZED_KERNELS}" == "1" ]]; then
    OPTIMIZED_KERNEL_ARGS=(--liger-kernels)
fi
if [[ "${TORCH_COMPILE}" == "1" ]]; then
    OPTIMIZED_KERNEL_ARGS+=(--compile)
fi

"${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NGPUS}" \
    "${MAIN_SCRIPT}" \
    --distributed-backend nccl \
    --experiment-name "${EXPERIMENT_NAME}" \
    \
    --dataset fineweb \
    --fineweb-source auto \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length "${SEQ_LEN}" \
    --streaming \
    --datasets-dir "${DATASETS_DIR}" \
    --workers 2 \
    \
    --model llama \
    --attention-type standard \
    --linear-parameterization tucker \
    --target-parameter-count "${TARGET_PARAMETER_COUNT}" \
    --target-parameter-tolerance "${TARGET_PARAMETER_TOLERANCE}" \
    --tucker-rank "${TUCKER_RANK}" \
    --tucker-forward-mode "${TUCKER_FORWARD_MODE}" \
    --tucker-contract-chunk-size "${TUCKER_CONTRACT_CHUNK_SIZE}" \
    --tucker-head-contract-chunk-size "${TUCKER_HEAD_CONTRACT_CHUNK_SIZE}" \
    --tucker-dense-adamw-matrices \
    --no-tucker-equal-params \
    --tucker-retract-every-step \
    --n-layer "${N_LAYER}" \
    --n-embd "${N_EMBD}" \
    --n-head "${N_HEAD}" \
    --multiple-of "${MULTIPLE_OF}" \
    --dtype bfloat16 \
    ${OPTIMIZED_KERNEL_ARGS[@]+"${OPTIMIZED_KERNEL_ARGS[@]}"} \
    ${CHECKPOINTING_ARG[@]+"${CHECKPOINTING_ARG[@]}"} \
    \
    --opt muon \
    --lr "${LR}" \
    --momentum "${MUON_MOMENTUM}" \
    --lite-muon-theta "${MUON_MOMENTUM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --beta1 "${ADAMW_BETA1}" \
    --beta2 "${ADAMW_BETA2}" \
    --grad-clip "${GRAD_CLIP}" \
    \
    --scheduler wsd \
    --warmup-steps "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --wsd-fract-decay "${WSD_FRACT_DECAY}" \
    --wsd-final-lr-scale "${WSD_FINAL_LR_SCALE}" \
    --decay-type "${DECAY_TYPE}" \
    \
    --batch-size "${BATCH_SIZE}" \
    --acc-steps "${ACC_STEPS}" \
    \
    --eval-interval 500 \
    --eval-batches 32 \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --downstream-eval-enabled \
    --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 2000 \
    --lm-eval-datasets wikitext103 \
    --log-interval 50 \
    --stable-rank-interval 50 \
    --spectrum-interval 1000 \
    --stable-rank-log-spectrum \
    \
    --latest-ckpt-interval 5000 \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    ${WANDB_ENTITY_ARG[@]+"${WANDB_ENTITY_ARG[@]}"} \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "1xChinchilla-tucker-retract" \
    --wandb-tags bf16 muon streaming standard-attention internal-tucker dense-lm-head qr-retract
