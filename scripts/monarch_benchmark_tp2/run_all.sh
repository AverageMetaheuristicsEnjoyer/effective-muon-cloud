#!/usr/bin/env bash
# Run the four TP=2 variants on an exclusive pair of GPUs.
set -euo pipefail

gpu_pair=${1:?usage: $0 GPU0,GPU1 [smoke|small|medium|large]}
geometry=${2:-small}
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
    --geometry "$geometry" --variant "$variant" \
    --warmup-steps 200 --measured-steps 500 \
    --output "results/monarch_tp2/${geometry}_${variant}.json" "${extra[@]}"
done
