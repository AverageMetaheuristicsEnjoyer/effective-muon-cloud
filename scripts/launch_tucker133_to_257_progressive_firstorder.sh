#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/single_gpu/tucker_transformer/fineweb_standard_attention_tensorion_muon_adamw_tucker_retract_1x_chinchilla_wd002.sh"

# The model starts from the validated 133M rank tuple.  The progressive
# controller chooses monotone per-module ranks nearest the three intermediate
# budgets and reaches full multilinear ranks at 257,676,352 parameters.
export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"llama257m_tucker_progressive_133m_to_257m_firstorder_calibrated_bs32acc4"}
export TARGET_PARAMETER_COUNT=133000000
export TARGET_PARAMETER_TOLERANCE=${TARGET_PARAMETER_TOLERANCE:-12312}
export TUCKER_RANKS="22,27,22,27"
export TUCKER_PROGRESSIVE_STAGES=${TUCKER_PROGRESSIVE_STAGES:-"0:133000000 4000:160000000 8000:190000000 12000:225000000 16000:257676352"}
export TUCKER_PROGRESSIVE_WARMUP_STEPS=${TUCKER_PROGRESSIVE_WARMUP_STEPS:-400}
export TUCKER_PROGRESSIVE_SEED=${TUCKER_PROGRESSIVE_SEED:-1701}
export TUCKER_PROGRESSIVE_VERIFY_RTOL=${TUCKER_PROGRESSIVE_VERIFY_RTOL:-5e-5}

# Shape-changing Parameters cannot be used with DDP buckets, even at world size
# one.  The base launcher therefore invokes plain Python for this experiment.
export SINGLE_PROCESS=1
export TUCKER_VECTOR_TRANSPORT=1
export TUCKER_RIEMANNIAN_MUON=1
export TUCKER_DENSE_ADAMW_MATRICES=1
export TUCKER_LR_SCALING_MODE=first_order_calibrated
export TUCKER_LR_SCALING_POST_NS_PROJECT=0
export TUCKER_LR_SCALING_USE_STIEFEL_UNIT_NORM=1
export WEIGHT_DECAY=0.1
export WEIGHT_DECAY_TAG=wd01

export ITERATIONS=${ITERATIONS:-39250}
export WARMUP=${WARMUP:-2000}
export BATCH_SIZE=${BATCH_SIZE:-32}
export ACC_STEPS=${ACC_STEPS:-4}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
export EVAL_BATCHES=${EVAL_BATCHES:-16}
export SAVE_BEST_VAL_CHECKPOINT=1
# Shape-aware latest checkpoints are intentionally frequent enough to make a
# shared-server interruption inexpensive.
export LATEST_CKPT_INTERVAL=${LATEST_CKPT_INTERVAL:-500}

exec bash "${BASE_LAUNCHER}"
