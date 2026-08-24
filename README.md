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

### Streaming from Hugging Face (no local download)

To skip the download and stream the same parquet shards directly from HF, pass
`--streaming --fineweb-source hf` to the training command (defaults target
`sample/100BT`):

```bash
--streaming --fineweb-source hf \
    --fineweb-hf-repo-id HuggingFaceFW/fineweb-edu \
    --fineweb-hf-data-prefix sample/100BT
```

`--fineweb-source auto` (the default) uses local shards when present and falls
back to HF streaming otherwise. The revision is pinned to a commit SHA, so the
stream is byte-for-byte identical to the local shards and fully reproducible
across runs with the same `--data-seed`.

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

## Tensorized Transformer (arXiv:1906.09777)

The project can replace every standard attention block with the Multi-linear
Attention from [A Tensorized Transformer for Language Modeling](https://arxiv.org/abs/1906.09777).
It uses one shared set of Q/K/V projections plus trainable diagonal BTD cores.

Run the WikiText-103/core-2 setup through the ready-made script:

```bash
bash scripts/single_gpu/tensorized_transformer/paper_wikitext103.sh
```

The script downloads/tokenizes WikiText-103 on first use and defaults to the
parameter-matched reconstruction configuration selected for this project:
256 dimensions, 6 layers, sequence length 80, rank 314, and core-2. It logs to
W&B in the same format as the scripts from `ruslan_logs`: train/validation
loss and perplexity, LR, throughput, gradient norm, GPU memory, per-matrix
normalized stable rank, and singular-value line plots. Useful overrides include:

```bash
WANDB_PROJECT=my-project WANDB_ENTITY=my-team \
BATCH_SIZE=16 ACC_STEPS=4 \
bash scripts/single_gpu/tensorized_transformer/paper_wikitext103.sh 1
```

Tensorized-attention flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--attention-type tensorized` | `standard` globally | Enable Multi-linear Attention. |
| `--tensorized-mode split_concat` | optional | Paper Eq. (8), with cubic sequence complexity. |
| `--tensorized-mode reconstruction` | script default | Corollary-1 quadratic-memory form selected by the supplied script. |
| `--tensorized-rank` | `314` in script | Shared Q/K/V factor rank; the global CLI default remains the paper's 40. |
| `--tensorized-num-cores` | `2` | Number of averaged diagonal BTD cores (`core-2`). |
| `--tensorized-query-chunk-size` | `8` | Bounds peak memory in `split_concat` mode without changing its result. |
| `--tensorized-causal` | on | Prevent future-token leakage during next-token training. |
| `--no-tensorized-causal` | off | Reproduce the public author code's unmasked behavior exactly. |

To reproduce the paper's exact rank-40 SplitConcat setup instead of the
parameter-matched default:

```bash
TENSORIZED_MODE=split_concat TENSORIZED_RANK=40 \
bash scripts/single_gpu/tensorized_transformer/paper_wikitext103.sh
```

Run the FineWeb reconstruction model with plain Muon, pure Tucker-parameterised
Linear layers, and the same 257M/WSD/warmup-2000/logging setup as the MuonBP
control experiment. Tensorized-attention rank is 1023; Tucker rank is a
separate hyperparameter and defaults to the rank-only match `259`:

```bash
WANDB_ENTITY="efficient-muon" \
WANDB_PROJECT="muon-variations" \
EXPERIMENT_NAME="tensorized_reconstruction_r1023_tucker_r259_pure_muon_lr1e-3_1xC" \
bash scripts/single_gpu/tensorized_transformer/fineweb_reconstruction_muon_wsd.sh 1
```

Every one of the Llama model's 85 independent `Linear` modules is replaced:
Q/K/V/O, gate/up/down in all 12 blocks, and `lm_head`. With the default scalar
rank `259`, each mode rank is capped by its tensor dimension:

| Linear shape | Resolved Tucker ranks | Parameters vs. dense per layer |
|--------------|------------------------|--------------------------------|
| `1024 -> 1023` | `(32,32,31,33)` | `+4,098` |
| `1024 -> 1024` | `(32,32,32,32)` | `+4,096` |
| `1024 -> 2816` | `(32,32,44,64)` | `+8,080` |
| `2816 -> 1024` | `(44,64,32,32)` | `+8,080` |
| `1024 -> 50304` | `(32,32,192,259)` | `-483,054` |

The resulting pure-Tucker model has `257,181,058` parameters: `7,806` fewer
than the original `257,188,864` standard-attention control, inside the allowed
`12,312` tolerance. No sparse correction, residual, filler, or dummy parameter
is created; ranks alone determine the count. Neighbouring scalar ranks do not
fit the tolerance: rank 258 gives `256,984,188`, while rank 260 gives
`257,377,928`. Stable-rank and spectrum metrics reconstruct each effective
Tucker matrix, and FLOP metrics use its actual contractions.

Tucker controls:

| CLI flag / script variable | Meaning |
|----------------------------|---------|
| `--linear-parameterization tucker` | Replace every independent Llama `Linear`. |
| `--target-parameter-count 257188864` / `TARGET_PARAMETER_COUNT=...` | Validation target in pure mode; it does not alter the model. |
| `--target-parameter-tolerance 12312` / `TARGET_PARAMETER_TOLERANCE=...` | Allowed rank-only difference from the target. |
| `--tucker-rank 259` / `TUCKER_RANK=259` | Default single scalar rank; capped independently by every tensor mode. |
| `--tucker-rank auto` / `TUCKER_RANK=auto` | Pick a separate largest under-dense rank tuple for each shape. |
| `--tucker-rank 20` / `TUCKER_RANK=20` | Use one scalar rank, capped by each tensor mode. |
| `--tucker-ranks 20,20,20,20` / `TUCKER_RANKS=...` | Explicit four mode ranks (overrides scalar policy). |
| `--tucker-forward-mode auto` / `TUCKER_FORWARD_MODE=auto` | Avoid large token-wise intermediates during full-sequence training (recommended). |
| `--no-tucker-equal-params` / `TUCKER_EQUAL_PARAMS=0` | Pure rank-only Tucker, with no correction (script default). |
| `--tucker-equal-params` / `TUCKER_EQUAL_PARAMS=1` | Optional legacy exact-count mode using a trainable correction. |

Rank 259 is a global parameter-count match, not compression of every individual
matrix: Q/K/V/O and MLP mode ranks saturate, while most of the reduction comes
from `lm_head`. Relative to the dense rank-1023 tensorized model, the pure
Tucker model is 4,506 parameters larger. `auto` forward mode materialises the
effective weight for large batches, avoiding multi-gigabyte token-wise
intermediates in the Tucker `lm_head`; use `contract` only for deliberately low
ranks or small batches.

### One-GPU 1x Chinchilla with Tucker gauge fixing

The dedicated one-GPU run applies a QR retraction after every optimizer step:
factor columns become orthonormal and each triangular factor is absorbed into
the Tucker core, leaving every effective dense weight unchanged.

```bash
bash \
  scripts/single_gpu/tucker_transformer/fineweb_standard_attention_muon_tucker_retract_1x_chinchilla.sh
```

The run uses a 257M Llama, standard attention, pure Tucker rank 259, Muon,
BF16 autocast, sequence length 1024, microbatch 16, accumulation 8, and 39,250
steps. This is 5,144,576,000 training tokens, approximately 20 tokens per
parameter. W&B receives train/validation metrics, GPU memory, throughput,
stable-rank/spectrum diagnostics, and Tucker factor orthogonality errors.
Only the rotating `ckpts/latest` resume checkpoint is retained locally by
default.

`split_concat` is intended for the 30-100 token contexts used in the paper.
For 1K+ token experiments, use `TENSORIZED_MODE=reconstruction`; materializing
the paper's full third-order token tensor at that length is not practical.

## MuonBP (block-periodic Muon)

MuonBP ([arXiv:2510.16981](https://arxiv.org/abs/2510.16981)) is a Muon variant
that orthogonalizes each weight matrix **block-by-block** on most steps and does
a **full** Newton-Schulz orthogonalization only every `P`-th step. The two
stepsizes (block vs. full) are not two numbers you tune — they come from the
paper's dimension-based RMS scaling, so you only set one base `--lr` (the
"full" learning rate) and the block-step learning rate follows automatically.

Run it (257M, 1x Chinchilla, single GPU):

```bash
bash scripts/single_gpu/fineweb/baselines/1xChinchilla/muonbp_lr1e-3.sh
# arg = #GPUs, e.g. `... muonbp_lr1e-3.sh 2` (must divide 128)
```

The plain-Muon control keeps the same data, model, token budget, WSD schedule,
evaluation, checkpoints, and W&B stable-rank/spectrum logging:

```bash
bash scripts/single_gpu/fineweb/baselines/1xChinchilla/muon_lr1e-3.sh
```

Select the optimizer with `--opt muonbp`. Its parameters (with defaults):

| Flag                        | Default | Meaning                                                                 |
|-----------------------------|---------|-------------------------------------------------------------------------|
| `--muonbp-period`           | `5`     | Full orthogonalization every N steps; block-orthogonalize on the others. `5` is the paper's recommended value. `N<=1` = plain Muon (full every step). |
| `--muonbp-nblocks`          | `2`     | Number of contiguous blocks each matrix is split into on a block step (paper: number of matrix shards / TP degree). `2` mirrors the paper's small-model TP=2 setting. |
| `--lr`                      | `1e-3`  | Base ("full") learning rate. The effective block-step LR follows from the dimension-based RMS scaling — no separate value to set. |
| `--momentum`                | `0.95`  | Momentum for the Muon heavy-ball buffer.                                |
| `--weight-decay`            | `0.1`   | Applied to the AdamW group only (1-D params, embeddings, lm_head), as in Muon. |
| `--beta1` / `--beta2` / `--eps` | `0.9` / `0.99` / `1e-7` | AdamW hyperparameters for the embeddings / lm_head / 1-D group. |

MuonBP builds its own Muon/AdamW parameter split internally (2-D projection
matrices → block-periodic Muon; everything else → AdamW), so it ignores the
usual `--model monarch` path. Use it with the standard `--model llama`.

## Tensorion

Tensorion ([arXiv:2606.25975](https://arxiv.org/abs/2606.25975)) generalizes
Muon's matrix update to higher-order tensors. Select it with `--opt tensorion`.
For every eligible tensor, the optimizer selects the fixed offline unfolding
from Eq. (24), orthogonalizes its momentum matrix, folds the update back, and
applies Algorithm 1's `0.2*sqrt(max(m_tau,n_tau))` scaling. Eligible 2-D
matrices use the standard Nesterov Muon update, while the remaining parameters
use AdamW in the same scheduler-compatible optimizer object.

In Tucker models, `core_matrix` remains physically 2-D for checkpoint and
forward compatibility, but Tensorion sees it as the logical 4-D core
`[r3,r4,r1,r2]`. With the default `--tensorion-min-dim 3`, eligible block
Tucker cores use Tensorion, Tucker factor matrices use Muon, and norms,
embeddings, plus the excluded `lm_head` use AdamW. Setting
`--tensorion-min-dim 2` routes eligible matrices through Tensorion's matrix
case, which is mathematically Muon.

| Flag | Default | Meaning |
|------|---------|---------|
| `--tensorion-min-dim` | `3` | Minimum logical tensor order handled by Tensorion. |
| `--tensorion-ns-steps` | `5` | Newton-Schulz iterations. |
| `--tensorion-orthogonalization` | `ns` | `ns` for the practical update or `svd` for the exact LMO. |
| `--tensorion-nesterov` | off | Optional Nesterov variant; the paper algorithm uses heavy-ball momentum. |
| `--no-tensorion-adjust-lr` | off | Disable the paper's unfolding-dimension step scaling. |
| `--momentum` | `0.95` | Tensorion momentum coefficient. |

### Stable-rank / spectrum logging

To reproduce the `stable_rank`-style W&B metrics (normalized stable rank per
projection group, following the NuMuon plots), add:

| Flag                         | Default | Meaning                                                                  |
|------------------------------|---------|--------------------------------------------------------------------------|
| `--stable-rank-interval`     | `0`     | Log normalized stable rank every N iterations (`0` = disabled).          |
| `--stable-rank-log-spectrum` | off     | Also log a singular-value line plot for every tracked block/group.       |

Exactly as in `ruslan_logs`, five relative-depth blocks are tracked. Every
Q/K/V/O/up/gate/down matrix gets its own scalar key, for example
`stable_rank/block00_qkvo_q_proj`; no value is averaged across layers or
matrices. With `--stable-rank-log-spectrum`, each `(block, group)` also gets a
`spectrum/blockXX_*` multi-line plot. Every snapshot is retained as a separate
curve in that same chart, so at step 5000 it contains the curves from steps
1000, 2000, ..., 5000. Q/K/V/O singular-value vectors are concatenated without
averaging into the qkvo group plot. The FineWeb Tucker script enables this
every 1000 iterations, producing snapshots at 1000, 2000, ..., 39000.

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
