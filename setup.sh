#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

YELLOW='\033[1;33m'
GREEN='\033[1;32m'
RED='\033[1;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn() { echo -e "${YELLOW}[setup]${NC} $*"; }
die()  { echo -e "${RED}[setup]${NC} $*" >&2; exit 1; }

command -v python >/dev/null 2>&1 || die "python not found on PATH"
command -v pip    >/dev/null 2>&1 || die "pip not found on PATH"
command -v nvcc   >/dev/null 2>&1 || die "nvcc not found on PATH — needed to detect the CUDA wheel slot"

PY_RAW="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_IS_FREE_THREADED="$(python -c 'import sysconfig; print(int(sysconfig.get_config_var("Py_GIL_DISABLED") or 0))')"
[ "$PY_IS_FREE_THREADED" = "0" ] || die "Free-threaded Python detected (Py_GIL_DISABLED=1). \
No flash-attn 2 wheels exist for ${PY_RAW}t — use a regular CPython build."
case "$PY_RAW" in
    3.10|3.11|3.12|3.13|3.14) ;;
    *) die "Python $PY_RAW is outside torch 2.9 support (need 3.10..3.14)." ;;
esac
PY_TAG="cp${PY_RAW//./}"

HOST_CUDA="$(nvcc --version | grep -oE 'release [0-9]+\.[0-9]+' | awk '{print $2}')"
[ -n "$HOST_CUDA" ] || die "Could not parse CUDA version from nvcc output."
HOST_CUDA_MAJOR="${HOST_CUDA%%.*}"
HOST_CUDA_MINOR="${HOST_CUDA##*.}"
HOST_CUDA_INT=$(( HOST_CUDA_MAJOR * 100 + HOST_CUDA_MINOR ))

# torch 2.9 ships wheels only for cu126 / cu128 / cu130; snap host CUDA down.
if   [ "$HOST_CUDA_INT" -ge 1300 ]; then CUDA_COMPACT="130"
elif [ "$HOST_CUDA_INT" -ge 1208 ]; then CUDA_COMPACT="128"
elif [ "$HOST_CUDA_INT" -ge 1206 ]; then CUDA_COMPACT="126"
else
    die "Host CUDA $HOST_CUDA is < 12.6 — torch 2.9 doesn't support it. \
Upgrade CUDA or pick a different torch."
fi

log "Detected Python $PY_RAW ($PY_TAG), host CUDA $HOST_CUDA -> wheel slot cu$CUDA_COMPACT"

log "Upgrading pip / setuptools / wheel"
pip install --upgrade pip setuptools wheel

TORCH_VERSION="${TORCH_VERSION:-2.9.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.24.1}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu${CUDA_COMPACT}}"

if python -c "import torch, torchvision; assert torch.__version__.startswith('${TORCH_VERSION%.*}'); assert torchvision.__version__.startswith('${TORCHVISION_VERSION%.*}')" 2>/dev/null; then
    log "torch ${TORCH_VERSION%.*}.x + torchvision ${TORCHVISION_VERSION%.*}.x already installed — skipping"
else
    log "Installing torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} from ${TORCH_INDEX}"
    pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "${TORCH_INDEX}"
fi

FA_VERSION="${FA_VERSION:-2.8.3}"
FA_RELEASE_TAG="${FA_RELEASE_TAG:-v0.7.16}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FA_RELEASE_TAG}/flash_attn-${FA_VERSION}+cu${CUDA_COMPACT}torch2.9-${PY_TAG}-${PY_TAG}-linux_x86_64.whl}"

if python -c "import flash_attn" 2>/dev/null; then
    log "flash_attn already installed — skipping"
else
    log "Resolved flash-attn wheel: $FLASH_ATTN_WHL"
    if command -v curl >/dev/null 2>&1; then
        if ! curl -fsIL "$FLASH_ATTN_WHL" >/dev/null; then
            die "Wheel URL not reachable. Check Python/CUDA support, or override FLASH_ATTN_WHL. \
See https://mjunya.com/flash-attention-prebuild-wheels/"
        fi
    fi
    pip install "$FLASH_ATTN_WHL"
fi

TORCHAO_VERSION="${TORCHAO_VERSION:-0.15.0}"
if python -c "import torchao" 2>/dev/null; then
    log "torchao already installed — skipping"
else
    log "Installing torchao==${TORCHAO_VERSION}"
    pip install "torchao==${TORCHAO_VERSION}"
fi

log "Installing requirements.txt"
pip install -r requirements.txt

log "Verifying imports"
python - <<'PY'
import torch, flash_attn
print(f"  torch          {torch.__version__}")
print(f"  flash_attn     {flash_attn.__version__}")
print(f"  cuda available {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device         {torch.cuda.get_device_name(0)}")
PY

log "Done."
echo
echo "Next steps:"
echo "  export HF_TOKEN=<your_hf_token>"
echo "  export WANDB_API_KEY=<your_wandb_api_key>"
echo "  bash scripts/single_gpu/fineweb/baselines/4xChinchilla/adamw_lr1e-3.sh"
