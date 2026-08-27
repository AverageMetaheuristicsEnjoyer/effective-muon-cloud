#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

NGPUS=1
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-29500}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"muon-variations"}
WANDB_ENTITY=${WANDB_ENTITY:-""}
DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"llama257m_tucker_r259_tensorion_muon_adamw_ns6_retract_1x_chinchilla_wd002_evalbs16"}

N_LAYER=12
N_EMBD=1024
N_HEAD=8
SEQ_LEN=1024
MULTIPLE_OF=256
TARGET_PARAMETER_COUNT=${TARGET_PARAMETER_COUNT:-257188864}
TARGET_PARAMETER_TOLERANCE=${TARGET_PARAMETER_TOLERANCE:-12312}
TUCKER_RANK=259
TUCKER_RANKS=${TUCKER_RANKS:-""}
TUCKER_ATTENTION_RANKS=${TUCKER_ATTENTION_RANKS:-""}
TUCKER_GATE_UP_RANKS=${TUCKER_GATE_UP_RANKS:-""}
TUCKER_DOWN_RANKS=${TUCKER_DOWN_RANKS:-""}
TUCKER_FORWARD_MODE=${TUCKER_FORWARD_MODE:-auto}
TRAIN_ENTRY=${TRAIN_ENTRY:-src/main.py}
TUCKER_VECTOR_TRANSPORT=${TUCKER_VECTOR_TRANSPORT:-0}
TUCKER_RIEMANNIAN_MUON=${TUCKER_RIEMANNIAN_MUON:-0}
TUCKER_RIEMANNIAN_MUON_POST_NS_PROJECT=${TUCKER_RIEMANNIAN_MUON_POST_NS_PROJECT:-0}
TUCKER_DENSE_ADAMW_MATRICES=${TUCKER_DENSE_ADAMW_MATRICES:-0}
TUCKER_LR_SCALING_MODE=${TUCKER_LR_SCALING_MODE:-none}
TUCKER_LR_SCALING_EPS=${TUCKER_LR_SCALING_EPS:-1e-8}
TUCKER_LR_SCALING_POWER_ITERS=${TUCKER_LR_SCALING_POWER_ITERS:-1}
TUCKER_LR_SCALING_USE_STIEFEL_UNIT_NORM=${TUCKER_LR_SCALING_USE_STIEFEL_UNIT_NORM:-1}
TUCKER_LR_SCALING_POST_NS_PROJECT=${TUCKER_LR_SCALING_POST_NS_PROJECT:-1}
TUCKER_LR_SCALING_STIEFEL_DRIFT_THRESHOLD=${TUCKER_LR_SCALING_STIEFEL_DRIFT_THRESHOLD:-1e-3}
TUCKER_LR_SCALING_STRICT_BOUND_CHECK=${TUCKER_LR_SCALING_STRICT_BOUND_CHECK:-0}
TUCKER_LR_SCALING_EXACT_SVD_DEBUG=${TUCKER_LR_SCALING_EXACT_SVD_DEBUG:-0}
TUCKER_LR_SCALING_LOG_INTERVAL=${TUCKER_LR_SCALING_LOG_INTERVAL:-100}
SAVE_BEST_VAL_CHECKPOINT=${SAVE_BEST_VAL_CHECKPOINT:-0}
LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-5000}
ITERATIONS=${ITERATIONS:-39250}
BATCH_SIZE=${BATCH_SIZE:-16}
ACC_STEPS=${ACC_STEPS:-8}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-16}
EVAL_BATCHES=${EVAL_BATCHES:-32}
LOG_INTERVAL=${LOG_INTERVAL:-50}
SINGLE_PROCESS=${SINGLE_PROCESS:-0}
TUCKER_PROGRESSIVE_STAGES=${TUCKER_PROGRESSIVE_STAGES:-""}
TUCKER_PROGRESSIVE_WARMUP_STEPS=${TUCKER_PROGRESSIVE_WARMUP_STEPS:-400}
TUCKER_PROGRESSIVE_SEED=${TUCKER_PROGRESSIVE_SEED:-1701}
TUCKER_PROGRESSIVE_VERIFY_RTOL=${TUCKER_PROGRESSIVE_VERIFY_RTOL:-5e-5}
RESUME_FROM=${RESUME_FROM:-""}

LR=1e-3
MOMENTUM=0.95
ADAMW_BETA1=0.9
ADAMW_BETA2=0.99
WEIGHT_DECAY=${WEIGHT_DECAY:-0.02}
WEIGHT_DECAY_TAG=${WEIGHT_DECAY_TAG:-"wd002"}
GRAD_CLIP=1.0
WARMUP=${WARMUP:-2000}
WSD_FRACT_DECAY=0.1
WSD_FINAL_LR_SCALE=0.0
DECAY_TYPE=cosine

WANDB_ENTITY_ARG=()
if [[ -n "${WANDB_ENTITY}" ]]; then
    WANDB_ENTITY_ARG=(--wandb-entity "${WANDB_ENTITY}")
fi

TUCKER_TRANSPORT_ARGS=()
TUCKER_RANK_ARGS=(--tucker-rank "${TUCKER_RANK}")
TUCKER_LR_SCALING_ARGS=(
    --tucker-lr-scaling-mode "${TUCKER_LR_SCALING_MODE}"
    --tucker-lr-scaling-eps "${TUCKER_LR_SCALING_EPS}"
    --tucker-lr-scaling-power-iters "${TUCKER_LR_SCALING_POWER_ITERS}"
    --tucker-lr-scaling-stiefel-drift-threshold "${TUCKER_LR_SCALING_STIEFEL_DRIFT_THRESHOLD}"
    --tucker-lr-scaling-log-interval "${TUCKER_LR_SCALING_LOG_INTERVAL}"
)
BEST_VAL_CHECKPOINT_ARGS=()
WANDB_TAGS=(
    bf16 tensorion muon adamw streaming standard-attention tucker qr-retract "${WEIGHT_DECAY_TAG}"
)
if [[ "${TUCKER_RIEMANNIAN_MUON}" == "1" && "${TUCKER_VECTOR_TRANSPORT}" != "1" ]]; then
    echo "TUCKER_RIEMANNIAN_MUON=1 requires TUCKER_VECTOR_TRANSPORT=1" >&2
    exit 2
fi
if [[ "${TUCKER_RIEMANNIAN_MUON_POST_NS_PROJECT}" == "1" && "${TUCKER_RIEMANNIAN_MUON}" != "1" ]]; then
    echo "TUCKER_RIEMANNIAN_MUON_POST_NS_PROJECT=1 requires TUCKER_RIEMANNIAN_MUON=1" >&2
    exit 2
fi
if [[ "${TUCKER_VECTOR_TRANSPORT}" == "1" ]]; then
    TUCKER_TRANSPORT_ARGS+=(--tucker-vector-transport)
    WANDB_TAGS+=(vector-transport)
fi
if [[ "${TUCKER_RIEMANNIAN_MUON}" == "1" ]]; then
    TUCKER_TRANSPORT_ARGS+=(--tucker-riemannian-muon)
    WANDB_TAGS+=(riemannian-muon stiefel-gradient-projection)
fi
if [[ "${TUCKER_RIEMANNIAN_MUON_POST_NS_PROJECT}" == "1" ]]; then
    TUCKER_TRANSPORT_ARGS+=(--tucker-riemannian-muon-post-ns-project)
    WANDB_TAGS+=(riemannian-muon-post-ns-tangent-projection)
fi
if [[ "${TUCKER_DENSE_ADAMW_MATRICES}" == "1" ]]; then
    TUCKER_TRANSPORT_ARGS+=(--tucker-dense-adamw-matrices)
    WANDB_TAGS+=(dense-adamw-matrices dense-lm-head)
fi
if [[ "${TUCKER_LR_SCALING_USE_STIEFEL_UNIT_NORM}" == "1" ]]; then
    TUCKER_LR_SCALING_ARGS+=(--tucker-lr-scaling-use-stiefel-unit-norm)
else
    TUCKER_LR_SCALING_ARGS+=(--no-tucker-lr-scaling-use-stiefel-unit-norm)
fi
if [[ "${TUCKER_LR_SCALING_POST_NS_PROJECT}" == "1" ]]; then
    TUCKER_LR_SCALING_ARGS+=(--tucker-lr-scaling-post-ns-project)
else
    TUCKER_LR_SCALING_ARGS+=(--no-tucker-lr-scaling-post-ns-project)
    WANDB_TAGS+=(no-post-ns-tangent-projection)
fi
if [[ "${TUCKER_LR_SCALING_STRICT_BOUND_CHECK}" == "1" ]]; then
    TUCKER_LR_SCALING_ARGS+=(--tucker-lr-scaling-strict-bound-check)
fi
if [[ "${TUCKER_LR_SCALING_EXACT_SVD_DEBUG}" == "1" ]]; then
    TUCKER_LR_SCALING_ARGS+=(--tucker-lr-scaling-exact-svd-debug)
fi
if [[ "${TUCKER_LR_SCALING_MODE}" != "none" ]]; then
    WANDB_TAGS+=("tucker-lr-${TUCKER_LR_SCALING_MODE}" "power-iters-${TUCKER_LR_SCALING_POWER_ITERS}")
fi
if [[ -n "${TUCKER_RANKS}" ]]; then
    TUCKER_RANK_ARGS=(--tucker-ranks "${TUCKER_RANKS}")
    WANDB_TAGS+=("tucker-ranks-${TUCKER_RANKS//,/x}")
fi
if [[ -n "${TUCKER_ATTENTION_RANKS}" ]]; then
    TUCKER_RANK_ARGS+=(--tucker-attention-ranks "${TUCKER_ATTENTION_RANKS}")
    WANDB_TAGS+=("tucker-attn-ranks-${TUCKER_ATTENTION_RANKS//,/x}")
fi
if [[ -n "${TUCKER_GATE_UP_RANKS}" ]]; then
    TUCKER_RANK_ARGS+=(--tucker-gate-up-ranks "${TUCKER_GATE_UP_RANKS}")
    WANDB_TAGS+=("tucker-gate-up-ranks-${TUCKER_GATE_UP_RANKS//,/x}")
fi
if [[ -n "${TUCKER_DOWN_RANKS}" ]]; then
    TUCKER_RANK_ARGS+=(--tucker-down-ranks "${TUCKER_DOWN_RANKS}")
    WANDB_TAGS+=("tucker-down-ranks-${TUCKER_DOWN_RANKS//,/x}")
fi
if [[ "${SAVE_BEST_VAL_CHECKPOINT}" == "1" ]]; then
    BEST_VAL_CHECKPOINT_ARGS+=(--save-best-val-checkpoint)
    WANDB_TAGS+=(best-val-checkpoint)
fi

TRAIN_COMMAND=(
    torchrun
    --nnodes=1
    --node_rank=0
    --master_addr="${MASTER_ADDR}"
    --master_port="${MASTER_PORT}"
    --nproc_per_node="${NGPUS}"
    "${TRAIN_ENTRY}"
)
DISTRIBUTED_ARGS=(--distributed-backend nccl)
PROGRESSIVE_ARGS=()
RESUME_ARGS=()
if [[ -n "${RESUME_FROM}" ]]; then
    RESUME_ARGS=(--resume-from "${RESUME_FROM}")
fi
if [[ -n "${TUCKER_PROGRESSIVE_STAGES}" ]]; then
    if [[ "${SINGLE_PROCESS}" != "1" ]]; then
        echo "TUCKER_PROGRESSIVE_STAGES requires SINGLE_PROCESS=1" >&2
        exit 2
    fi
    read -r -a PROGRESSIVE_STAGE_VALUES <<< "${TUCKER_PROGRESSIVE_STAGES}"
    PROGRESSIVE_ARGS=(
        --tucker-progressive-stages "${PROGRESSIVE_STAGE_VALUES[@]}"
        --tucker-progressive-warmup-steps "${TUCKER_PROGRESSIVE_WARMUP_STEPS}"
        --tucker-progressive-seed "${TUCKER_PROGRESSIVE_SEED}"
        --tucker-progressive-verify-rtol "${TUCKER_PROGRESSIVE_VERIFY_RTOL}"
    )
    WANDB_TAGS+=(progressive-rank-growth function-preserving-growth)
fi
if [[ "${SINGLE_PROCESS}" == "1" ]]; then
    TRAIN_COMMAND=(python "${TRAIN_ENTRY}")
    DISTRIBUTED_ARGS=()
fi

"${TRAIN_COMMAND[@]}" \
    ${DISTRIBUTED_ARGS[@]+"${DISTRIBUTED_ARGS[@]}"} \
    --experiment-name "${EXPERIMENT_NAME}" \
    ${RESUME_ARGS[@]+"${RESUME_ARGS[@]}"} \
    --dataset fineweb \
    --fineweb-source auto \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length "${SEQ_LEN}" \
    --streaming \
    --datasets-dir "${DATASETS_DIR}" \
    --workers 2 \
    --model llama \
    --attention-type standard \
    --linear-parameterization tucker \
    --target-parameter-count "${TARGET_PARAMETER_COUNT}" \
    --target-parameter-tolerance "${TARGET_PARAMETER_TOLERANCE}" \
    "${TUCKER_RANK_ARGS[@]}" \
    ${PROGRESSIVE_ARGS[@]+"${PROGRESSIVE_ARGS[@]}"} \
    --tucker-forward-mode "${TUCKER_FORWARD_MODE}" \
    --no-tucker-equal-params \
    --tucker-retract-every-step \
    ${TUCKER_TRANSPORT_ARGS[@]+"${TUCKER_TRANSPORT_ARGS[@]}"} \
    "${TUCKER_LR_SCALING_ARGS[@]}" \
    --n-layer "${N_LAYER}" \
    --n-embd "${N_EMBD}" \
    --n-head "${N_HEAD}" \
    --multiple-of "${MULTIPLE_OF}" \
    --dtype bfloat16 \
    --opt tensorion \
    --tensorion-min-dim 3 \
    --tensorion-ns-steps 6 \
    --tensorion-orthogonalization ns \
    --lr "${LR}" \
    --momentum "${MOMENTUM}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --beta1 "${ADAMW_BETA1}" \
    --beta2 "${ADAMW_BETA2}" \
    --grad-clip "${GRAD_CLIP}" \
    --scheduler wsd \
    --warmup-steps "${WARMUP}" \
    --iterations "${ITERATIONS}" \
    --wsd-fract-decay "${WSD_FRACT_DECAY}" \
    --wsd-final-lr-scale "${WSD_FINAL_LR_SCALE}" \
    --decay-type "${DECAY_TYPE}" \
    --batch-size "${BATCH_SIZE}" \
    --acc-steps "${ACC_STEPS}" \
    --eval-interval 500 \
    --eval-batches "${EVAL_BATCHES}" \
    --eval-batch-size "${EVAL_BATCH_SIZE}" \
    --downstream-eval-enabled \
    --downstream-eval-interval 2000 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 2000 \
    --lm-eval-datasets wikitext103 \
    --log-interval "${LOG_INTERVAL}" \
    --stable-rank-interval 50 \
    --spectrum-interval 1000 \
    --stable-rank-log-spectrum \
    --latest-ckpt-interval "${LATEST_CKPT_INTERVAL}" \
    ${BEST_VAL_CHECKPOINT_ARGS[@]+"${BEST_VAL_CHECKPOINT_ARGS[@]}"} \
    --results-base-folder "${RESULTS_DIR}" \
    --wandb \
    ${WANDB_ENTITY_ARG[@]+"${WANDB_ENTITY_ARG[@]}"} \
    --wandb-project "${WANDB_PROJECT}" \
    --wandb-group "1xChinchilla-tucker-retract" \
    --wandb-tags "${WANDB_TAGS[@]}"
