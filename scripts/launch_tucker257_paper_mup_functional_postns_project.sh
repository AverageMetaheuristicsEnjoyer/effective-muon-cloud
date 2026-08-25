#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_LAUNCHER="${SCRIPT_DIR}/single_gpu/tucker_transformer/fineweb_standard_attention_tensorion_muon_adamw_tucker_retract_1x_chinchilla_wd002.sh"

export EXPERIMENT_NAME=${EXPERIMENT_NAME:-"llama257m_tucker_r259_tensorion_riemannian_muon_paper_mup_functional_postns_tangent_adamw_ns6_vector_transport_densehead_retract_1x_chinchilla_wd01_bestval_trainbs32_acc4_evalbs32"}
export TARGET_PARAMETER_COUNT=${TARGET_PARAMETER_COUNT:-257676352}
export TARGET_PARAMETER_TOLERANCE=${TARGET_PARAMETER_TOLERANCE:-12312}
export TUCKER_RANKS=${TUCKER_RANKS:-""}
export TUCKER_VECTOR_TRANSPORT=1
export TUCKER_RIEMANNIAN_MUON=1
export TUCKER_DENSE_ADAMW_MATRICES=1
export TUCKER_LR_SCALING_MODE=paper_mup_functional
export TUCKER_LR_SCALING_POST_NS_PROJECT=1
export TUCKER_LR_SCALING_LOG_INTERVAL=${TUCKER_LR_SCALING_LOG_INTERVAL:-100}
export WEIGHT_DECAY=0.1
export WEIGHT_DECAY_TAG=wd01
export SAVE_BEST_VAL_CHECKPOINT=1
export LATEST_CKPT_INTERVAL=0
export BATCH_SIZE=${BATCH_SIZE:-32}
export ACC_STEPS=${ACC_STEPS:-4}
export EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-32}
export EVAL_BATCHES=${EVAL_BATCHES:-16}

exec bash "${BASE_LAUNCHER}"
