#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"

export MAIN_SCRIPT="${SCRIPT_DIR}/train_entry.py"
export TUCKER_CUSTOM_CACHE_POLICY=${TUCKER_CUSTOM_CACHE_POLICY:-hybrid_gate_up}
export TUCKER_ONLINE_CE=0
export TUCKER_PARALLEL_MUON=${TUCKER_PARALLEL_MUON:-1}
export TUCKER_MUON_CORE_MICROBATCH=${TUCKER_MUON_CORE_MICROBATCH:-1}
export TUCKER_MUON_STREAMS=${TUCKER_MUON_STREAMS:-2}
export TUCKER_GROUPED_SMALL_MUON=${TUCKER_GROUPED_SMALL_MUON:-0}
export TUCKER_GROUPED_RETRACTION=${TUCKER_GROUPED_RETRACTION:-1}

exec bash \
    "${ROOT}/scripts/single_gpu/tucker_transformer/fineweb_standard_attention_muon_tucker_retract_1x_chinchilla.sh"
