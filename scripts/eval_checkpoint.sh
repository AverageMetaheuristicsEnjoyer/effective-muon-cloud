#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: bash scripts/eval_checkpoint.sh --ckpt <dir> [options]

Required:
  --ckpt PATH          Checkpoint directory (contains main.pt)
  --n-layer N          Transformer layers
  --n-embd N           Embedding width
  --n-head N           Attention heads
  --batch-size N       Per-process micro-batch (training value)
  --acc-steps N        Gradient accumulation steps (training value)

Optional:
  --ngpus N            Number of GPUs (default: 1)
  --seq-len N          Sequence length (default: 1024)
  --multiple-of N      FFN dim multiple (default: 256)
  --wandb              Log results to W&B (off by default)

Environment overrides (rarely needed):
  DATASETS_DIR    FineWeb parquet shard dir   (default ./data/fineweb-edu/sample/100BT)
  EVAL_CACHE_DIR  Downstream-eval cache       (default ./evals_cache)
  RESULTS_DIR     Local log dir               (default ./exps)
  WANDB_PROJECT   W&B project name (with --wandb)  (default fp8-pretrain)
USAGE
}

CKPT=""
NGPUS=1
N_LAYER=""
N_EMBD=""
N_HEAD=""
SEQ_LEN=1024
MULTIPLE_OF=256
BATCH_SIZE=""
ACC_STEPS=""
USE_WANDB=0

while [ $# -gt 0 ]; do
    case "$1" in
        --ckpt)        CKPT="$2"; shift 2 ;;
        --ngpus)       NGPUS="$2"; shift 2 ;;
        --n-layer)     N_LAYER="$2"; shift 2 ;;
        --n-embd)      N_EMBD="$2"; shift 2 ;;
        --n-head)      N_HEAD="$2"; shift 2 ;;
        --seq-len)     SEQ_LEN="$2"; shift 2 ;;
        --multiple-of) MULTIPLE_OF="$2"; shift 2 ;;
        --batch-size)  BATCH_SIZE="$2"; shift 2 ;;
        --acc-steps)   ACC_STEPS="$2"; shift 2 ;;
        --wandb)       USE_WANDB=1; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

WANDB_ARGS=()
if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS=(--wandb --wandb-project "${WANDB_PROJECT:-fp8-pretrain}" --wandb-tags eval-only)
fi

missing=()
[ -n "$CKPT" ]       || missing+=("--ckpt")
[ -n "$N_LAYER" ]    || missing+=("--n-layer")
[ -n "$N_EMBD" ]     || missing+=("--n-embd")
[ -n "$N_HEAD" ]     || missing+=("--n-head")
[ -n "$BATCH_SIZE" ] || missing+=("--batch-size")
[ -n "$ACC_STEPS" ]  || missing+=("--acc-steps")
if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing required: ${missing[*]}" >&2
    usage
    exit 1
fi

DATASETS_DIR=${DATASETS_DIR:-"./data/fineweb-edu/sample/100BT"}
EVAL_CACHE_DIR=${EVAL_CACHE_DIR:-"./evals_cache"}
RESULTS_DIR=${RESULTS_DIR:-"./exps"}
WANDB_PROJECT=${WANDB_PROJECT:-"fp8-pretrain"}

torchrun --standalone --nproc_per_node="${NGPUS}" src/eval_checkpoint.py \
    --distributed-backend nccl \
    --experiment-name "$(basename "$(dirname "$(dirname "${CKPT}")")")" \
    --resume-from "${CKPT}" \
    \
    --dataset fineweb \
    --datasets-dir "${DATASETS_DIR}" \
    --eval-cache-dir "${EVAL_CACHE_DIR}" \
    --sequence-length ${SEQ_LEN} \
    --workers 8 \
    \
    --model llama \
    --n-layer ${N_LAYER} \
    --n-embd ${N_EMBD} \
    --n-head ${N_HEAD} \
    --multiple-of ${MULTIPLE_OF} \
    --dtype bfloat16 \
    \
    --opt adamw \
    --lr 1e-3 \
    --batch-size ${BATCH_SIZE} \
    --acc-steps ${ACC_STEPS} \
    \
    --eval-batches 32 \
    --downstream-eval-enabled \
    --downstream-eval-interval 1 \
    --downstream-task-group basic_v2 \
    --lm-eval-enabled \
    --lm-eval-interval 1 \
    --lm-eval-datasets wikitext103 \
    \
    --results-base-folder "${RESULTS_DIR}" \
    "${WANDB_ARGS[@]}"
