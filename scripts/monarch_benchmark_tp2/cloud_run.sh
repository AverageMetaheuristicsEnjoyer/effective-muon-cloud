#!/bin/bash
# Entry point for cloud.ru jobs launched through mlsub.
#
#   mlsub run --repo <public mirror> --branch tp2-bench --image torch28 --no-pip \
#     --entry scripts/monarch_benchmark_tp2/cloud_run.sh --gpus 2 \
#     --args "20 100 1024"
#
# First argument may instead be:
#   selftest   the cheap --gpus cpu rehearsal: environment probe, dependency
#              install, and the Gloo TP=2 correctness test
#   peek       print the newest log and the runs recorded so far
#
# A failed mlsub job shows no logs at all, so output is teed to the persistent
# workspace disk and this script always exits zero.
set -u

root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"

# /home/jovyan has reached its inode quota before; the workspace volumes had
# not, so take the first one that actually accepts a directory.
workspace=${TP2_WORKSPACE:-}
if [ -z "$workspace" ]; then
    for volume in /workspace-SR006.nfs3 /workspace-SR006.nfs2 /home/jovyan /tmp; do
        if mkdir -p "$volume/monarch-tp2/logs" 2>/dev/null; then
            workspace="$volume/monarch-tp2"
            break
        fi
    done
fi
mkdir -p "$workspace/logs" "$workspace/results"

# mlsub points PYTHONUSERBASE at /home/jovyan, which has used all 500 000 of
# its inodes, so pip install --user there fails with ENOSPC while df -h still
# reports free gigabytes.
export PYTHONUSERBASE="$workspace/userbase"
export PATH="$PYTHONUSERBASE/bin:$PATH"
mkdir -p "$PYTHONUSERBASE"

if [ "${1:-}" = "peek" ]; then
    echo "workspace: $workspace"
    echo "=== recorded runs ==="
    find "$workspace/results" -name '*.json' 2>/dev/null | sort || echo "none yet"
    newest=$(ls -t "$workspace"/logs/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-200}" "$newest"
    exit 0
fi

if [ "${1:-}" = "export" ]; then
    # NCCL_DEBUG=INFO floods the job log, so the numbers leave as one compact
    # line per variant rather than as their JSON payloads.
    python3 - "$workspace/results/monarch_tp2" <<'JSON'
import json
import sys
from pathlib import Path

COLUMNS = ("run", "variant", "step_ms", "forward_ms", "backward_ms", "optimizer_ms",
           "newton_schulz_ms", "activation_nccl_ms", "optimizer_nccl_ms",
           "activation_nccl_mb", "optimizer_nccl_mb", "peak_gb", "validation_err")

print("PT\t" + "\t".join(COLUMNS))
for path in sorted(Path(sys.argv[1]).glob("*/*.json")):
    payload = json.loads(path.read_text())
    median = payload["median"]
    error = payload.get("validation_max_abs_error")
    print("PT\t" + "\t".join([
        path.parent.name,
        payload["variant"],
        f"{median['step_ms']:.3f}",
        f"{median['forward_ms']:.3f}",
        f"{median['backward_ms']:.3f}",
        f"{median['optimizer_ms']:.3f}",
        f"{median['newton_schulz_ms']:.3f}",
        f"{median['activation_nccl_ms']:.3f}",
        f"{median['optimizer_nccl_ms']:.3f}",
        f"{median['activation_nccl_bytes'] / 1e6:.1f}",
        f"{median['optimizer_nccl_bytes'] / 1e6:.1f}",
        f"{median['peak_allocated_bytes'] / 1e9:.3f}",
        "" if error is None else f"{error:.2e}",
    ]))
    for key, entry in payload["collective_breakdown"].items():
        print(f"CB\t{payload['variant']}\t{key}\t{entry['calls']}\t"
              f"{entry['bytes']}\t{entry['ms']:.3f}")
JSON
    exit 0
fi

# --gpus 2 runs this entry point once per MPI rank on the same two-GPU node.
# Every rank but the first has to stand down, or each one launches its own
# torchrun and the two copies fight over the same pair of cards until OOM.
if [ "${OMPI_COMM_WORLD_RANK:-0}" != "0" ]; then
    echo "MPI rank ${OMPI_COMM_WORLD_RANK} stands down; rank 0 drives torchrun."
    exit 0
fi

LOG="$workspace/logs/$(date +%F_%H%M%S)-$$.log"

{
    echo "commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "python: $(python -V 2>&1)"
    echo "workspace: $workspace"
    echo "userbase: ${PYTHONUSERBASE:-unset}"
    df -h /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1
    df -i /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3 2>&1

    # The image ships torch only. megatron-core 0.16.1 calls newton_schulz_tp
    # with mode= and OrthogonalizedOptimizer with use_nesterov=, which is the
    # Emerging-Optimizers v0.1.0 API; v0.2.0 renamed both, and PyPI only
    # publishes 0.3.0, so this pin has to come from git.
    python -c "import megatron.core" 2>/dev/null \
        || pip install --user -q megatron-core==0.16.1
    python -c "import emerging_optimizers" 2>/dev/null \
        || pip install --user -q \
           "emerging-optimizers @ git+https://github.com/NVIDIA-NeMo/Emerging-Optimizers@v0.1.0"

    python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(),
      "devices:", torch.cuda.device_count(), flush=True)
for index in range(torch.cuda.device_count()):
    properties = torch.cuda.get_device_properties(index)
    print(f"  cuda:{index} {properties.name} {properties.total_memory} bytes", flush=True)
import megatron.core, emerging_optimizers
from megatron.core.optimizer.muon import HAVE_EMERGING_OPTIMIZERS, TensorParallelMuon
print("megatron-core:", megatron.core.__version__,
      "emerging-optimizers:", emerging_optimizers.__version__,
      "muon wired:", HAVE_EMERGING_OPTIMIZERS, flush=True)
PY

    # torchrun is not guaranteed to be on PATH in the container, and the
    # benchmark scripts want a single executable.
    printf '%s\n' '#!/bin/sh' 'exec python -m torch.distributed.run "$@"' > "$root/torchrun-shim"
    chmod +x "$root/torchrun-shim"
    export TORCHRUN="$root/torchrun-shim"

    if [ "${1:-}" = "selftest" ]; then
        "$TORCHRUN" --standalone --nproc_per_node=2 -m scripts.monarch_benchmark_tp2.test_tp2_cpu
        echo "SELFTEST_EXIT=$?"
    else
        export RESULTS_ROOT="$workspace/results"
        bash scripts/monarch_benchmark_tp2/run_llama7b_tp2_benchmark.sh 0,1 "$@"
        echo "BENCHMARK_EXIT=$?"
    fi
} 2>&1 | tee "$LOG"

echo "log: $LOG"
exit 0
