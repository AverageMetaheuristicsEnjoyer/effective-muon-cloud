#!/usr/bin/env bash
# Full 6.89B LLaMA TP=2 systems benchmark: four Muon communication modes.
set -euo pipefail

gpu_pair=${1:?usage: $0 GPU0,GPU1 [warmup_steps] [measured_steps] [sequence_length]}
warmup_steps=${2:-20}
measured_steps=${3:-100}
sequence_length=${4:-1024}

# Override these through the environment for an explicit hyperparameter sweep.
lr=${LR:-1e-4}
momentum=${MOMENTUM:-0.95}
beta1=${BETA1:-0.9}
beta2=${BETA2:-0.95}
weight_decay=${WEIGHT_DECAY:-0.1}
min_free_gib=${MIN_FREE_GIB:-65}
torchrun=${TORCHRUN:-.venv/bin/torchrun}

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

IFS=, read -r gpu0 gpu1 <<< "$gpu_pair"
if [[ -z "${gpu0:-}" || -z "${gpu1:-}" ]]; then
  echo "GPU pair must have the form GPU0,GPU1" >&2
  exit 2
fi

# Cloud containers hand out whole GPUs and ship no nvidia-smi, so there is
# nothing to admit against there.
if command -v nvidia-smi >/dev/null 2>&1; then
  for gpu in "$gpu0" "$gpu1"; do
    free_mib=$(nvidia-smi --id="$gpu" --query-gpu=memory.free --format=csv,noheader,nounits)
    if (( free_mib < min_free_gib * 1024 )); then
      echo "GPU $gpu has ${free_mib} MiB free; require ${min_free_gib} GiB." >&2
      exit 1
    fi
  done
else
  echo "nvidia-smi unavailable; skipping the ${min_free_gib} GiB admission check." >&2
fi

stamp=$(date -u +%Y%m%dT%H%M%SZ)
outdir="${RESULTS_ROOT:-results}/monarch_tp2/llama7b_tp2_${stamp}"
mkdir -p "$outdir"

printf '%s\n' \
  "gpu_pair=$gpu_pair" \
  "geometry=llama7b" \
  "layers=32" \
  "hidden=4096" \
  "heads=32" \
  "ffn_hidden=11008" \
  "sequence_length=$sequence_length" \
  "batch_size=1" \
  "warmup_steps=$warmup_steps" \
  "measured_steps=$measured_steps" \
  "lr=$lr" \
  "momentum=$momentum" \
  "beta1=$beta1" \
  "beta2=$beta2" \
  "weight_decay=$weight_decay" \
  "min_free_gib=$min_free_gib" > "$outdir/run_config.env"

for variant in dense_duplicated dense_distributed dense_blockwise monarch_muon; do
  extra=()
  if [[ "$variant" == "monarch_muon" ]]; then
    extra+=(--validate-monarch)
  fi
  echo "Starting $variant"
  CUDA_VISIBLE_DEVICES="$gpu_pair" CUDA_DEVICE_MAX_CONNECTIONS=1 \
    "$torchrun" --standalone --nproc_per_node=2 \
    -m scripts.monarch_benchmark_tp2.benchmark_tp2 \
    --geometry llama7b --tokens "$sequence_length" --layers 32 \
    --variant "$variant" --batch-size 1 \
    --warmup-steps "$warmup_steps" --measured-steps "$measured_steps" \
    --lr "$lr" --momentum "$momentum" --beta1 "$beta1" --beta2 "$beta2" \
    --weight-decay "$weight_decay" \
    --output "$outdir/${variant}.json" "${extra[@]}" 2>&1 | tee "$outdir/${variant}.log"
done

echo "Completed. JSON metrics and logs: $outdir"
