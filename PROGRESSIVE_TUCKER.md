# Progressive Tucker 133M → 257M

The run starts from the validated global Tucker ranks `(22, 27, 22, 27)` and
grows only the Tucker ranks. Hidden size, depth, token embeddings, norms, and
the dense language-model head do not change.

## Resolved schedule

For the current 12-layer Llama configuration the automatic proportional rank
solver resolves the default schedule as follows:

| Iteration | Requested | Actual | Attention | Gate/up | Down |
|---:|---:|---:|---|---|---|
| 0 | 133,000,000 | 132,990,448 | 22,27,22,27 | 22,27,22,27 | 22,27,22,27 |
| 4,000 | 160,000,000 | 159,965,968 | 25,29,25,29 | 25,29,30,40 | 30,40,25,29 |
| 8,000 | 190,000,000 | 190,252,624 | 28,30,28,30 | 28,30,35,50 | 35,50,28,30 |
| 12,000 | 225,000,000 | 224,635,072 | 30,31,30,31 | 30,31,41,58 | 41,58,30,31 |
| 16,000 | 257,676,352 | 257,676,352 | 32,32,32,32 | 32,32,44,64 | 44,64,32,32 |

The final ranks are the full mode sizes, so the final reconstructed matrices
have unrestricted dense expressivity.

At a transition, old factor columns are retained exactly, orthogonal columns
are appended, and the old core is copied into a zero-filled leading block.
The code verifies the reconstructed dense weight before continuing. Tensorion
and Muon momentum states are expanded with zeros, while their old blocks,
global scheduler, data-reader state, and non-Tucker AdamW state are preserved.

## Install the overlay

Copy this directory over the current experiment repository without deleting
files that are not present in the overlay:

```bash
rsync -a /path/to/spectron_patch/ /path/to/Tucker_tensorion_experiment/
```

## Test

From the experiment repository:

```bash
pytest -q tests/test_progressive_tucker.py tests/test_tucker_lr_scaling.py
```

## Launch

The progressive run must use one plain Python process. DDP caches parameter
shapes and is deliberately rejected, even for a world size of one.

```bash
cd /path/to/Tucker_tensorion_experiment
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_ENTITY=efficient-muon
export WANDB_MODE=online
bash scripts/launch_tucker133_to_257_progressive_firstorder.sh
```

Do not use a numerical CUDA ordinal on a shared server. Resolve and re-check
the physical UUID with `nvidia-smi -L` immediately before launch. The expected
peak is the same class as the existing 257M `batch_size=32` run, so require at
least 49,152 MiB free before starting and retain an 8,192 MiB reserve.

The default effective batch is `32 × 4 = 128`. Override only the physical
batch and accumulation together if needed:

```bash
BATCH_SIZE=16 ACC_STEPS=8 EVAL_BATCH_SIZE=16 \
  bash scripts/launch_tucker133_to_257_progressive_firstorder.sh
```

To resume, use the same launcher arguments and point it at the saved checkpoint:

```bash
RESUME_FROM=/path/to/experiment/ckpts/latest \
  bash scripts/launch_tucker133_to_257_progressive_firstorder.sh
```

The checkpoint contains the active rank plan, and parameters are resized before
model and optimizer state loading.
