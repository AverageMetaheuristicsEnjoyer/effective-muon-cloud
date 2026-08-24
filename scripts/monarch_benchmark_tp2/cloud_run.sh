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

if [ "${1:-}" = "peek" ]; then
    echo "workspace: $workspace"
    echo "=== recorded runs ==="
    find "$workspace/results" -name '*.json' 2>/dev/null | sort || echo "none yet"
    newest=$(ls -t "$workspace"/logs/*.log 2>/dev/null | head -1)
    echo "=== tail of ${newest:-no log} ==="
    [ -n "$newest" ] && tail -"${2:-200}" "$newest"
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
