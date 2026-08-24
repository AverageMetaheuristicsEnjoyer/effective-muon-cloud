#!/usr/bin/env bash
# One TP=2 LLaMA-like kill-test point: all four optimizer variants.
set -euo pipefail

gpu_pair=${1:?usage: $0 GPU0,GPU1 [small|medium|large|llama7b] [tokens] [layers] [warmup] [measured]}
geometry=${2:-large}
tokens=${3:-2048}
layers=${4:-4}
warmup=${5:-200}
measured=${6:-500}
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

mkdir -p results/monarch_tp2
for variant in dense_duplicated dense_distributed dense_blockwise monarch_muon; do
  extra=()
  if [[ "$variant" == "monarch_muon" ]]; then
    extra+=(--validate-monarch)
  fi
  CUDA_VISIBLE_DEVICES="$gpu_pair" CUDA_DEVICE_MAX_CONNECTIONS=1 \
    .venv/bin/torchrun --standalone --nproc_per_node=2 \
    -m scripts.monarch_benchmark_tp2.benchmark_tp2 \
    --geometry "$geometry" --tokens "$tokens" --layers "$layers" \
    --variant "$variant" --warmup-steps "$warmup" --measured-steps "$measured" \
    --output "results/monarch_tp2/${geometry}_t${tokens}_l${layers}_${variant}.json" \
    "${extra[@]}"
done
