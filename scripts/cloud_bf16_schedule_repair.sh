#!/usr/bin/env bash
set -u

mode=${1:-setup}
arm=${2:-}
root=$(cd "$(dirname "$0")/.." && pwd)
persist=${PERSIST_ROOT:-/tmp/efficient-training-bf16-repairs}
run_key=${REPAIR_RUN_KEY:-20260829}
rank=${OMPI_COMM_WORLD_RANK:-0}
local_rank=${OMPI_COMM_WORLD_LOCAL_RANK:-0}
world_size=${OMPI_COMM_WORLD_SIZE:-1}
log_dir=$persist/logs
state_dir=$persist/state
mkdir -p "$log_dir" "$state_dir"
log=$log_dir/${mode}-${arm:-none}-${run_key}-rank${rank}.log
export HF_HOME=${HF_HOME:-$persist/cache/huggingface}
export TORCH_HOME=${TORCH_HOME:-$persist/cache/torch}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$persist/cache/triton}
export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$persist/cache/pip}
export HF_SOURCE_REPO_ID=${HF_SOURCE_REPO_ID:-moderntalker/efficient_pretrain_checkpoints}
export HF_UPLOAD_REPO_ID=${HF_UPLOAD_REPO_ID:-AverageMetaheuristicsEnjoyer/efficient_pretrain_checkpoints}
export PYTHONPATH=$root/src${PYTHONPATH:+:$PYTHONPATH}

report_disks() {
    echo "=== DISKS ==="
    for path in /tmp /home/jovyan /workspace-SR006.nfs2 /workspace-SR006.nfs3; do
        if [[ -e $path ]]; then
            df -h "$path" | tail -1
            df -i "$path" | tail -1
        else
            echo "ABSENT $path"
        fi
    done
}

prepare_runtime() {
    local venv=$persist/venv
    local marker=$state_dir/runtime-${run_key}.ok
    local failed=$state_dir/runtime-${run_key}.failed
    if [[ $rank == 0 ]]; then
        rm -f "$marker" "$failed"
        python -m venv --system-site-packages "$venv" && \
            "$venv/bin/python" -m pip install --upgrade-strategy only-if-needed -r "$root/requirements.cloudru.txt" || \
            touch "$failed"
        [[ ! -f $failed ]] && touch "$marker"
    fi
    wait_for_file "$marker" "$failed" || return 1
    source "$venv/bin/activate"
}

setup_runtime() {
    cd "$root"
    report_disks
    prepare_runtime || return 1
    python - <<'PY'
import platform
import torch
import bitsandbytes
import datasets
import huggingface_hub
import pyarrow
import tensorly
import tiktoken
import transformers
import wandb
from distributed_shampoo import DistributedShampoo
import main

print("python=", platform.python_version())
print("torch=", torch.__version__)
print("torch_cuda=", torch.version.cuda)
print("wandb=", wandb.__version__)
print("efficient_training_import=PASS")
PY
}

map_mpi_env() {
    : "${OMPI_COMM_WORLD_RANK:?Cloud MPI rank is missing}"
    : "${OMPI_COMM_WORLD_LOCAL_RANK:?Cloud MPI local rank is missing}"
    : "${OMPI_COMM_WORLD_SIZE:?Cloud MPI world size is missing}"
    export RANK=$OMPI_COMM_WORLD_RANK
    export LOCAL_RANK=$OMPI_COMM_WORLD_LOCAL_RANK
    export WORLD_SIZE=$OMPI_COMM_WORLD_SIZE
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    export MASTER_PORT=${MASTER_PORT:-29500}
}

nccl_smoke() {
    map_mpi_env
    python - <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["RANK"])
local_rank = int(os.environ["LOCAL_RANK"])
world_size = int(os.environ["WORLD_SIZE"])
torch.cuda.set_device(local_rank)
dist.init_process_group("nccl")
value = torch.tensor([rank + 1.0], device=f"cuda:{local_rank}")
dist.all_reduce(value)
expected = world_size * (world_size + 1) / 2
assert value.item() == expected, (value.item(), expected)
print(
    f"nccl_identity=PASS rank={rank} local_rank={local_rank} world_size={world_size} "
    f"device={torch.cuda.get_device_name(local_rank)} sm={torch.cuda.get_device_capability(local_rank)}",
    flush=True,
)
dist.barrier()
dist.destroy_process_group()
PY
}

wait_for_file() {
    local wanted=$1
    local failed=$2
    for _ in $(seq 1 720); do
        [[ -f $wanted ]] && return 0
        [[ -f $failed ]] && return 1
        sleep 5
    done
    return 1
}

download_source() {
    local remote_path=$1
    local expected_world=$2
    local source_root=$persist/hf-source
    local source_dir=$source_root/$remote_path
    local marker=$state_dir/download-${arm}-${run_key}.ok
    local failed=$state_dir/download-${arm}-${run_key}.failed

    if [[ $rank == 0 ]]; then
        rm -f "$marker" "$failed"
        ET_SOURCE_ROOT=$source_root ET_REMOTE_PATH=$remote_path ET_EXPECTED_WORLD=$expected_world \
            python - <<'PY' || touch "$failed"
import os
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(os.environ["ET_SOURCE_ROOT"])
remote = os.environ["ET_REMOTE_PATH"]
world = int(os.environ["ET_EXPECTED_WORLD"])
snapshot_download(
    repo_id=os.environ["HF_SOURCE_REPO_ID"],
    repo_type="model",
    allow_patterns=[f"{remote}/**"],
    local_dir=root,
)
source = root / remote
expected = [source / "main.pt"] + [source / f"worker_{rank}.pt" for rank in range(world)]
missing = [str(path) for path in expected if not path.is_file() or path.stat().st_size == 0]
if missing:
    raise RuntimeError(f"checkpoint download is incomplete: {missing}")
print(f"SOURCE_READY path={source} bytes={sum(path.stat().st_size for path in expected)}", flush=True)
PY
        [[ ! -f $failed ]] && touch "$marker"
    fi

    wait_for_file "$marker" "$failed" || return 1
}

upload_and_verify() {
    local checkpoint_dir=$1
    local remote_path=$2
    ET_CHECKPOINT_DIR=$checkpoint_dir ET_UPLOAD_PATH=$remote_path python - <<'PY'
import os
import shutil
from pathlib import Path
from huggingface_hub import HfApi

local = Path(os.environ["ET_CHECKPOINT_DIR"])
remote = os.environ["ET_UPLOAD_PATH"]
repo = os.environ["HF_UPLOAD_REPO_ID"]
local_files = {
    str(path.relative_to(local)): path.stat().st_size
    for path in sorted(local.rglob("*"))
    if path.is_file()
}
if not local_files:
    raise RuntimeError(f"checkpoint is empty: {local}")

api = HfApi(token=os.environ["HF_TOKEN"])
api.upload_folder(
    repo_id=repo,
    repo_type="model",
    folder_path=str(local),
    path_in_repo=remote,
    commit_message=f"Upload corrected BF16 schedule checkpoint {remote}",
)
remote_files = {}
for entry in api.list_repo_tree(repo, path_in_repo=remote, repo_type="model", recursive=True):
    size = getattr(entry, "size", None)
    if size is not None:
        remote_files[entry.path[len(remote) + 1:]] = size
mismatch = {name: size for name, size in local_files.items() if remote_files.get(name) != size}
if mismatch:
    raise RuntimeError(f"HF verification failed for {remote}: {mismatch}")
print(f"HF_VERIFIED path={repo}/{remote} files={len(local_files)} bytes={sum(local_files.values())}", flush=True)
shutil.rmtree(local.parent.parent)
print(f"LOCAL_RESULT_REMOVED path={local.parent.parent}", flush=True)
PY
}

run_repair() {
    map_mpi_env
    if [[ $mode == run ]]; then
        : "${HF_TOKEN:?HF_TOKEN is required}"
        : "${WANDB_API_KEY:?WANDB_API_KEY is required}"
    fi
    export WANDB_BASE_URL=${WANDB_BASE_URL:-https://wandb-radfan.ru}
    export WANDB_ENTITY=${WANDB_ENTITY:-andrey}
    export WANDB_PROJECT=${WANDB_PROJECT:-fp8-pretrain}
    export WANDB_MODE=online
    prepare_runtime || return 1

    local optimizer start target expected_world lr weight_decay beta2 batch_size acc_steps source_group source_name
    case $arm in
        galore-1c|galore-2c)
            optimizer=galore_adamw
            lr=1e-3
            weight_decay=0.1
            beta2=0.999
            source_group=4C_2gpus
            source_name=fineweb_galore_adamw_lr1e-3_wd0.1_bf16_grad_clip1e0_4C
            ;;
        slim-adam-1c|slim-adam-2c)
            optimizer=slim_adam
            lr=5e-4
            weight_decay=0.1
            beta2=0.999
            source_group=4C_2gpus
            source_name=fineweb_slim_adam_lr5e-4_wd0.1_bf16_grad_clip1e0_4C
            ;;
        shampoo-1c|shampoo-2c)
            optimizer=shampoo
            lr=1e-2
            weight_decay=0.01
            beta2=0.99
            source_group=4C_8gpus
            source_name=fineweb_shampoo_lr1e-2_wd0.01_bf16_grad_clip1e0_4C
            ;;
        *)
            echo "unknown repair arm: $arm" >&2
            return 2
            ;;
    esac

    case $arm in
        *-1c) start=35325; target=39250; budget=1xC ;;
        *-2c) start=70650; target=78500; budget=2xC ;;
    esac
    expected_world=4
    if [[ $optimizer == shampoo ]]; then
        batch_size=16
        acc_steps=8
    else
        batch_size=32
        acc_steps=4
    fi
    local warmup_steps=2000
    local decay_fraction=0.1
    local resume_args=()
    local smoke=0
    if [[ $mode == resume-smoke ]]; then
        smoke=1
        if [[ $optimizer == shampoo ]]; then
            expected_world=8
            batch_size=16
            acc_steps=8
        else
            expected_world=2
            batch_size=32
            acc_steps=4
        fi
        warmup_steps=0
        decay_fraction=1
        target=$((start + 2))
        local source_remote=intermediate-checkpoints/$source_group/$source_name/inter-ckpt-$start
        local source_dir=$persist/hf-source/$source_remote
        download_source "$source_remote" "$expected_world" || return 3
        resume_args=(--resume-from "$source_dir" --decay-from-checkpoint)
    else
        start=0
    fi
    [[ $world_size == "$expected_world" ]] || {
        echo "ARM_WORLD_SIZE_MISMATCH arm=$arm expected=$expected_world actual=$world_size" >&2
        return 2
    }

    report_disks
    nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader
    echo "SOURCE_COMMIT=$(git -C "$root" rev-parse HEAD)"
    echo "ARM_CONFIG arm=$arm optimizer=$optimizer start=$start target=$target world_size=$world_size schedule=wsd warmup=$warmup_steps fract_decay=$decay_fraction decay_type=cosine"

    local experiment=bf16_250m_${arm//-/_}_wsd_repair_${run_key}
    [[ $smoke == 1 ]] && experiment=smoke_${experiment}
    local group=bf16_250m_schedule_repairs
    local results_root=${CHECKPOINT_PERSIST_ROOT:-/home/jovyan/bf16-schedule-repair-checkpoints}/results
    local checkpoint_dir=$results_root/$group/$experiment/ckpts/latest
    local app_state=$state_dir/$experiment
    rm -f "$app_state".done.* "$app_state".failed "$app_state".uploaded
    mkdir -p "$results_root" "$persist/datasets" "$persist/evals-cache"

    local optimizer_args=()
    if [[ $optimizer == galore_adamw ]]; then
        optimizer_args=(--density 0.25 --update_gap 50 --proj_side std --proj_type svd --proj_params_lr_scale 1 --reset_statistics)
    elif [[ $optimizer == shampoo ]]; then
        optimizer_args=(--shampoo_beta3 -1 --shampoo_preconditioner_frequency 100 --shampoo_max_preconditioner_dim 1024)
    fi

    local eval_args=(
        --eval-interval 500
        --eval-batches 32
        --downstream-eval-enabled
        --downstream-eval-interval 2000
        --downstream-task-group basic_v2
        --lm-eval-enabled
        --lm-eval-interval 2000
        --lm-eval-datasets wikitext103
        --log-interval 50
    )
    local output_args=(
        --latest-ckpt-interval "$target"
        --results-base-folder "$results_root"
        --wandb
        --wandb-project "$WANDB_PROJECT"
        --wandb-group "$group"
        --wandb-tags bf16 schedule-repair 250m "$budget" "$optimizer" cloudru
    )
    if [[ $smoke == 1 ]]; then
        eval_args=(--eval-interval 1 --eval-batches 1 --log-interval 1)
        output_args=(--no-local-save)
    fi

    cd "$root"
    python src/main.py \
        --distributed-backend nccl \
        --experiment-name "$experiment" \
        "${resume_args[@]}" \
        --dataset fineweb \
        --fineweb-source hf \
        --fineweb-hf-repo-id HuggingFaceFW/fineweb-edu \
        --fineweb-hf-data-prefix sample/100BT \
        --fineweb-hf-revision 87f09149ef4734204d70ed1d046ddc9ca3f2b8f9 \
        --datasets-dir "$persist/datasets" \
        --eval-cache-dir "$persist/evals-cache" \
        --sequence-length 1024 \
        --streaming \
        --workers 8 \
        --model llama \
        --n-layer 12 \
        --n-embd 1024 \
        --n-head 8 \
        --multiple-of 256 \
        --dtype bfloat16 \
        --opt "$optimizer" \
        --lr "$lr" \
        --weight-decay "$weight_decay" \
        --beta1 0.9 \
        --beta2 "$beta2" \
        --eps 1e-7 \
        --grad-clip 1.0 \
        "${optimizer_args[@]}" \
        --scheduler wsd \
        --warmup-steps "$warmup_steps" \
        --iterations "$target" \
        --wsd-fract-decay "$decay_fraction" \
        --wsd-final-lr-scale 0 \
        --decay-type cosine \
        --batch-size "$batch_size" \
        --acc-steps "$acc_steps" \
        "${eval_args[@]}" \
        "${output_args[@]}"
    local app_code=$?
    echo "TRAIN_EXIT=$app_code arm=$arm rank=$rank"
    if [[ $app_code -eq 0 ]]; then
        touch "$app_state.done.$rank"
    else
        touch "$app_state.failed"
    fi

    if [[ $rank == 0 ]]; then
        for _ in $(seq 1 720); do
            [[ -f $app_state.failed ]] && break
            done_count=$(find "$state_dir" -maxdepth 1 -name "$(basename "$app_state").done.*" -type f | wc -l)
            [[ $done_count -eq $world_size ]] && break
            sleep 5
        done
        done_count=$(find "$state_dir" -maxdepth 1 -name "$(basename "$app_state").done.*" -type f | wc -l)
        if [[ ! -f $app_state.failed && $done_count -eq $world_size && $smoke == 1 ]]; then
            touch "$app_state.uploaded"
        elif [[ ! -f $app_state.failed && $done_count -eq $world_size ]]; then
            local upload_path=intermediate-checkpoints/$group/$experiment/inter-ckpt-$target
            if upload_and_verify "$checkpoint_dir" "$upload_path"; then
                touch "$app_state.uploaded"
            else
                touch "$app_state.failed"
            fi
        fi
    fi

    wait_for_file "$app_state.uploaded" "$app_state.failed" || return 4
    report_disks
    echo "ARM_COMPLETE arm=$arm"
}

(
    case $mode in
        setup) setup_runtime ;;
        nccl-smoke) nccl_smoke ;;
        resume-smoke) run_repair ;;
        run) run_repair ;;
        *) echo "usage: $0 setup | nccl-smoke | resume-smoke ARM | run ARM" >&2; exit 2 ;;
    esac
) >"$log" 2>&1
code=$?

echo "APP_EXIT=$code mode=$mode arm=${arm:-none} rank=$rank"
echo "LOG=$log"
tail -n 200 "$log"
exit 0
