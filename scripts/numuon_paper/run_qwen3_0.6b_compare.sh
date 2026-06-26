#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
NGPUS=${1:-${NGPUS:-8}}
OPTS=${OPTS:-"adamw muon numuon"}

for opt in ${OPTS}; do
    echo "===== ${opt} ====="
    bash "${SCRIPT_DIR}/run_qwen3_0.6b.sh" "${opt}" "${NGPUS}"
done
