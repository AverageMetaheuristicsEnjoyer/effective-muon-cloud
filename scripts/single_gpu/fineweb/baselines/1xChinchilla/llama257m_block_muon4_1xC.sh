#!/usr/bin/env bash
# 257M dense LLaMA with four independent row-wise Muon blocks per 2D weight.
set -euo pipefail

gpu=${1:?usage: $0 GPU_INDEX}

results_dir=${RESULTS_DIR:-"./exps"}
datasets_dir=${DATASETS_DIR:-"/data/datasets/fineweb-edu-100BT/sample/100BT"}
experiment_name=${EXPERIMENT_NAME:-"llama257m_block_muon4_1xC"}
iterations=${ITERATIONS:-39250}
warmup_steps=${WARMUP_STEPS:-2000}
permanent_ckpt_interval=${PERMANENT_CKPT_INTERVAL:-2000}

export WANDB_ENTITY=${WANDB_ENTITY:-"efficient-muon"}

CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/torchrun --standalone --nproc_per_node=1 src/main.py \
  --distributed-backend nccl \
  --experiment-name "$experiment_name" \
  --dataset fineweb \
  --eval-cache-dir "${EVAL_CACHE_DIR:-./evals_cache}" \
  --sequence-length 1024 \
  --streaming \
  --datasets-dir "$datasets_dir" \
  --workers 8 \
  --model llama \
  --n-layer 12 \
  --n-embd 1024 \
  --n-head 8 \
  --multiple-of 256 \
  --dtype bfloat16 \
  --opt muon \
  --muon-num-splits 4 \
  --muon-split-dim 0 \
  --lite-muon-theta 0.95 \
  --lite-ns-steps 5 \
  --lr 1e-3 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --grad-clip 1.0 \
  --scheduler wsd \
  --warmup-steps "$warmup_steps" \
  --iterations "$iterations" \
  --wsd-fract-decay 0.1 \
  --wsd-final-lr-scale 0.0 \
  --decay-type cosine \
  --batch-size 16 \
  --acc-steps 8 \
  --eval-interval 500 \
  --eval-batches 32 \
  --downstream-eval-enabled \
  --downstream-eval-interval 2000 \
  --downstream-task-group basic_v2 \
  --lm-eval-enabled \
  --lm-eval-interval 2000 \
  --lm-eval-datasets wikitext103 \
  --log-interval 50 \
  --latest-ckpt-interval "$permanent_ckpt_interval" \
  --permanent-ckpt-interval "$permanent_ckpt_interval" \
  --results-base-folder "$results_dir" \
  --wandb \
  --wandb-project muon-variations \
  --wandb-group 1xChinchilla \
  --wandb-tags baseline bf16 muon block_muon blocks4 streaming
