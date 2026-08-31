#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_monarch_1b_benchmark.sh ARM BLOCKS}
blocks=${2:?usage: cloud_monarch_1b_benchmark.sh ARM BLOCKS}

export RANK=${OMPI_COMM_WORLD_RANK:?missing OMPI_COMM_WORLD_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29533}
export PYTHONPATH=.:src

case "$arm" in
  adamw) optimizer=(--opt adamw --apply-monarch) ;;
  muon) optimizer=(--opt monarch_muon) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

gpu_uuid=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=uuid --format=csv,noheader)
echo "MONARCH_TRAIN_PROCESS rank=$RANK world_size=$WORLD_SIZE local_rank=$LOCAL_RANK pid=$$ gpu_uuid=$gpu_uuid"

python src/main.py \
  --distributed-backend nccl \
  --experiment-name "benchmark_1b_${arm}_monarch_n${blocks}_${WORLD_SIZE}g" \
  --dataset shakespeare-char \
  --datasets-dir "/tmp/monarch-1b-benchmark-${RANK}" \
  --model llama \
  --n-layer 20 \
  --n-embd 2048 \
  --n-head 16 \
  --intermediate-size 5632 \
  --multiple-of 256 \
  --vocab-size 50304 \
  --sequence-length 1024 \
  --dtype bfloat16 \
  "${optimizer[@]}" \
  --monarch-nblocks "$blocks" \
  --lr 1e-3 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --scheduler none \
  --warmup-steps 0 \
  --iterations 12 \
  --batch-size 16 \
  --acc-steps 8 \
  --eval-batch-size 16 \
  --eval-interval 12 \
  --eval-batches 1 \
  --log-interval 1 \
  --no-local-save
