#!/usr/bin/env bash
set -euo pipefail

export TUCKER_LATE_ROOT=${TUCKER_ORDER3_ROOT:-/workspace-SR006.nfs3/tucker-order3-pretrain-20260901}
export WANDB_MODE=${WANDB_MODE:-online}
exec bash scripts/cloud_progressive_late_run.sh "$@"
