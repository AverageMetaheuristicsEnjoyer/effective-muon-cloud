#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

exec bash \
    "${SCRIPT_DIR}/scripts/single_gpu/tucker_transformer/fineweb_standard_attention_muon_tucker_retract_1x_chinchilla.sh"
