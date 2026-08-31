#!/usr/bin/env bash
set -euo pipefail

arm=${1:?usage: cloud_monarch_1b_pretrain.sh adamw|muon BLOCKS}
blocks=${2:?usage: cloud_monarch_1b_pretrain.sh adamw|muon BLOCKS}

[[ ${MLSUB_IMAGE:-} == torch28 ]] || { echo "requires --image torch28" >&2; exit 2; }
[[ -z ${TORCHELASTIC_RUN_ID:-} ]] || { echo "nested torchrun is not allowed" >&2; exit 2; }
[[ $blocks == 2 || $blocks == 4 ]] || { echo "BLOCKS must be 2 or 4" >&2; exit 2; }

export RANK=${OMPI_COMM_WORLD_RANK:?missing OMPI_COMM_WORLD_RANK}
export WORLD_SIZE=${OMPI_COMM_WORLD_SIZE:?missing OMPI_COMM_WORLD_SIZE}
export LOCAL_RANK=${OMPI_COMM_WORLD_LOCAL_RANK:?missing OMPI_COMM_WORLD_LOCAL_RANK}
export MASTER_ADDR=${MASTER_ADDR:-$(hostname)}
export MASTER_PORT=${MASTER_PORT:-29535}
export PYTHONPATH=.:src
export WANDB_ENTITY=${WANDB_ENTITY:-efficient-muon}

case "$WORLD_SIZE" in 1|2|4) ;; *) echo "WORLD_SIZE must be 1, 2, or 4" >&2; exit 2 ;; esac
case "$arm" in
  adamw) optimizer=(--opt adamw --apply-monarch) ;;
  muon) optimizer=(--opt monarch_muon) ;;
  *) echo "unknown arm: $arm" >&2; exit 2 ;;
esac

datasets_dir=${DATASETS_DIR:-/home/jovyan/data/fineweb-edu/sample/100BT}
results_dir=${RESULTS_DIR:-/workspace-SR006.nfs3/effective-muon-checkpoints/monarch-1b}
results_mount=${RESULTS_MOUNT:-/workspace-SR006.nfs3}
iterations=${ITERATIONS:-156891}
min_free_gb=${MIN_FREE_GB:-40}

[[ -d $datasets_dir ]] || { echo "missing dataset directory: $datasets_dir" >&2; exit 2; }
find "$datasets_dir" -maxdepth 1 -type f -name '*.parquet' -print -quit | grep -q . || {
  echo "no parquet shards in $datasets_dir" >&2
  exit 2
}
available_kb=$(df -Pk "$results_mount" | awk 'NR==2 {print $4}')
(( available_kb >= min_free_gb * 1024 * 1024 )) || {
  echo "need at least ${min_free_gb} GiB free on $results_mount" >&2
  df -h "$results_mount"
  exit 2
}
mkdir -p "$results_dir"

gpu_uuid=$(nvidia-smi -i "$LOCAL_RANK" --query-gpu=uuid --format=csv,noheader)
echo "MONARCH_TRAIN_PROCESS rank=$RANK world_size=$WORLD_SIZE local_rank=$LOCAL_RANK pid=$$ gpu_uuid=$gpu_uuid nested_torchrun=false"

exec python src/main.py \
  --distributed-backend nccl \
  --experiment-name "llama1b_${arm}_monarch_n${blocks}_1xC" \
  --dataset fineweb \
  --sequence-length 1024 \
  --streaming \
  --datasets-dir "$datasets_dir" \
  --workers 8 \
  --model llama \
  --n-layer 16 \
  --n-embd 2048 \
  --n-head 16 \
  --intermediate-size 5632 \
  --multiple-of 256 \
  --vocab-size 50304 \
  --dtype bfloat16 \
  "${optimizer[@]}" \
  --monarch-nblocks "$blocks" \
  --lr 1e-3 \
  --weight-decay 0.1 \
  --beta1 0.9 \
  --beta2 0.99 \
  --eps 1e-8 \
  --momentum 0.95 \
  --nesterov \
  --grad-clip 1.0 \
  --scheduler wsd \
  --warmup-steps 2000 \
  --iterations "$iterations" \
  --wsd-fract-decay 0.1 \
  --wsd-final-lr-scale 0.0 \
  --decay-type cosine \
  --batch-size 8 \
  --acc-steps 16 \
  --eval-batch-size 8 \
  --eval-interval 500 \
  --eval-batches 32 \
  --log-interval 50 \
  --latest-ckpt-interval 2000 \
  --results-base-folder "$results_dir" \
  --auto-resume \
  --wandb \
  --wandb-project "${WANDB_PROJECT:-muon-variations}" \
  --wandb-group 1xChinchilla_monarch_1b \
  --wandb-tags bf16 monarch "nblocks${blocks}" "$arm"
