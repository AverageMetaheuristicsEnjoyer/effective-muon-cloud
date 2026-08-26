#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

echo "run_fused_online_tucker.sh is deprecated: target lm_head is Dense." >&2
echo "Starting run_fused_dense_head_tucker.sh instead." >&2
exec bash "${SCRIPT_DIR}/run_fused_dense_head_tucker.sh"
