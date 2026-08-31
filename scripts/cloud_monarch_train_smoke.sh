#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_monarch_train_smoke.sh ARM BLOCKS}
blocks=${2:?usage: cloud_monarch_train_smoke.sh ARM BLOCKS}

export RANK=${OMPI_COMM_WORLD_RANK:?missing OMPI_COMM_WORLD_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29532}
export PYTHONPATH=.:src

case "$arm" in
  adamw) optimizer=(--opt adamw --apply-monarch) ;;
  muon) optimizer=(--opt monarch_muon) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

python src/main.py \
  --distributed-backend nccl \
  --experiment-name "smoke_${arm}_monarch_n${blocks}" \
  --dataset shakespeare-char \
  --datasets-dir "/tmp/monarch-smoke-${RANK}" \
  --model llama \
  --n-layer 2 \
  --n-embd 128 \
  --n-head 4 \
  --multiple-of 64 \
  --vocab-size 128 \
  --sequence-length 64 \
  --dtype bfloat16 \
  "${optimizer[@]}" \
  --monarch-nblocks "$blocks" \
  --lr 1e-3 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --scheduler none \
  --warmup-steps 0 \
  --iterations 2 \
  --batch-size 2 \
  --acc-steps 2 \
  --eval-batch-size 2 \
  --eval-interval 2 \
  --eval-batches 1 \
  --log-interval 1 \
  --no-local-save
