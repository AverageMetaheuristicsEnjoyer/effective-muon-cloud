#!/usr/bin/env bash
# Short TP=2 measurement/instrumentation check for a shared GPU pair.
set -euo pipefail

gpu_pair=${1:?usage: $0 GPU0,GPU1}
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
    --geometry smoke --variant "$variant" \
    --warmup-steps 3 --measured-steps 5 \
    --output "results/monarch_tp2/smoke_${variant}.json" "${extra[@]}"
done
