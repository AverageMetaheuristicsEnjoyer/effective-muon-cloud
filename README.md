# Stage 2

## TL;DR
```bash
conda env create -f environment.yml && conda activate huawei-stage2 && bash setup.sh
```

## Prerequisites

- Linux with NVIDIA GPU(s)
- Python 3.10 – 3.14 (no free-threaded `t` builds)
- A working `nvcc` on PATH — used by `setup.sh` to detect the CUDA wheel slot

```bash
nvcc --version      # confirm CUDA 12.6+ (12.8 / 13.0 also supported)
```

If unavaliable try to search for cuda-*/ inside /usr/local/lib or /usr/lib on your machine. Then 

```bash
export PATH=/usr/local/cuda-*/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-*/lib64:$LD_LIBRARY_PATH
```

## HuggingFace & W&B credentials

`hf download` needs `HF_TOKEN`. Training also logs to W&B:

```bash
export HF_TOKEN=<your_hf_token>
export WANDB_BASE_URL=<your_wandb_base_url>   # optional, defaults to https://api.wandb.ai
export WANDB_API_KEY=<your_wandb_api_key>
```

You can change wandb project in scripts or by env variable as well.

## Dataset

Training reads FineWeb-edu (`sample/100BT`) from local parquet shards. The
download is **not** part of `setup.sh` — run it once yourself before launching
any training script:

```bash
hf download HuggingFaceFW/fineweb-edu \
    --repo-type dataset \
    --include "sample/100BT/*" \
    --local-dir data/fineweb-edu
```

This drops parquet shards under `data/fineweb-edu/sample/100BT/`. The
scripts default `DATASETS_DIR=./data/fineweb-edu/sample/100BT` — it must
point directly at the directory containing the `*.parquet` files. See
[Scripts to run](#scripts-to-run) for the list of variables.

## Installation

The fastest path is the `setup.sh` script at the repo root. It detects your
Python and CUDA and picks the matching wheels automatically:

```bash
# inside your venv
bash setup.sh
```

Supported auto-detection range (everything else: follow the manual steps
below and pass overrides as env vars):

| Component | Supported              | Override env var                          |
|-----------|------------------------|-------------------------------------------|
| Python    | 3.10 – 3.14 (no `t`)   | (install matching CPython)                |
| CUDA      | 12.6 / 12.8 / 13.0     | `TORCH_INDEX`, `FLASH_ATTN_WHL`           |
| Torch     | 2.9.1                  | `TORCH_VERSION`, `TORCHVISION_VERSION`    |
| flash-attn| 2.8.3 (tag `v0.7.16`)  | `FA_VERSION`, `FA_RELEASE_TAG`, `FLASH_ATTN_WHL` |
| torchao   | 0.15.0                 | `TORCHAO_VERSION`                         |

Host CUDA is snapped down to the nearest supported wheel slot (e.g. CUDA
12.9 host → `cu128` wheels). If anything is out of range the script aborts
*before* installing anything wrong, with a message naming the override var.

### Using conda / miniconda (optional)

Alternative to the venv path: conda provides Python and a full CUDA toolkit
(including `nvcc`) inside the env, then `setup.sh` does the pip part unchanged.

```bash
conda env create -f environment.yml
conda activate huawei-stage2
bash setup.sh
```

`environment.yml` pins the CUDA toolkit version (default 12.8), which `setup.sh`
reads from `nvcc` to choose the wheel slot (`cu128`). Edit that pin for a
different slot (`12.6` → `cu126`, `13.0` → `cu130`); your GPU driver must
support the version you pick.

If you prefer to install everything by hand, follow the steps below.

### 1. PyTorch

Install `torch==2.9.1` together with the matching `torchvision==0.24.1`:

```bash
pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu130
```

### 2. flash-attn

Pick the wheel that matches your Python / torch / CUDA combination from
https://mjunya.com/flash-attention-prebuild-wheels/ and install it directly,
e.g.

```bash
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.8.3+cu130torch2.9-cp311-cp311-linux_x86_64.whl
```

### 3. torchao

Choose the `torchao` release that matches your `torch` version
(see https://github.com/pytorch/ao/issues/2919). For `torch==2.9.1`:

```bash
pip install torchao==0.15.0
```

### 4. Remaining Python deps

```bash
pip install -r requirements.txt
```

### Verification

```bash
python -c "import torch; import flash_attn; print('ok')"
```

## Scripts to run

Two model sizes are included, with one script per optimizer (AdamW, SOAP,
AdEMAMix, Muon).

- **257M, single GPU:** `scripts/single_gpu/fineweb/baselines/4xChinchilla/`
- **0.5B, 2 GPUs:** `scripts/multiple_gpus/0.5B/baselines/1xChinchilla_2gpus/`

For each script, just invoke it directly:

```bash
bash scripts/single_gpu/fineweb/baselines/4xChinchilla/adamw_lr1e-3.sh
bash scripts/multiple_gpus/0.5B/baselines/1xChinchilla_2gpus/adamw_lr1e-3.sh 2   # arg = #GPUs
```

You can run both model sizes with more GPUs. It scales batch size and accumulation steps automatically. Though the number of GPUs should devide 128. We did run such experiments and they didn't show much difference in performance with baselines.

Each script reads the following env vars (with the defaults shown):

| Variable          | Default               | Purpose                                  |
|-------------------|-----------------------|------------------------------------------|
| `DATASETS_DIR`    | `./data/fineweb-edu/sample/100BT` | Directory containing FineWeb-edu `*.parquet` shards |
| `EVAL_CACHE_DIR`  | `./evals_cache`       | Cache for downstream evaluation data     |
| `RESULTS_DIR`     | `./exps`              | Where W&B-shadow logs are written        |
| `WANDB_PROJECT`   | `fp8-pretrain`        | W&B project name                         |

We usually just edit `DATASETS_DIR` while everything else stays the same.

## Evaluate a checkpoint

To run the full eval suite (val loss/ppl/acc on FineWeb-edu, wikitext-103
perplexity, downstream task group `basic_v2`) on a saved checkpoint, use `scripts/eval_checkpoint.sh`.

The wrapper takes the checkpoint dir (must contain `main.pt`) plus the
model-shape flags from the training script that produced it. Each training
script writes `ckpts/latest/main.pt` every 5000 iterations and again at the
final iteration, so point `--ckpt` at that directory.

**257M (4xChinchilla)**

```bash
bash scripts/eval_checkpoint.sh \
    --ckpt   ./exps/4xChinchilla/<run-name>/ckpts/latest \
    --n-layer 12 --n-embd 1024 --n-head 8 \
    --batch-size 32 --acc-steps 4
```

**0.5B (1xChinchilla_2gpus)**

```bash
bash scripts/eval_checkpoint.sh \
    --ckpt   ./exps/1xChinchilla_2gpus/<run-name>/ckpts/latest \
    --n-layer 18 --n-embd 1280 --n-head 20 \
    --batch-size 16 --acc-steps 8 \
    --ngpus 2
```

Again, `--ngpus` can be any divisor of `batch-size × acc-steps` (= 128 for both
configs). Pass `--wandb` to also log the eval metrics to W&B (off by default) and don't forget to specify project for wandb.

## Credits

- [Quartet-II codebase](https://github.com/IST-DASLab/Quartet-II/) — training infrastructure baseline
- [COAT: Compressing Optimizer states and Activation for Memory-Efficient FP8 Training](https://arxiv.org/abs/2410.19313) — method baseline and training infrastructure
