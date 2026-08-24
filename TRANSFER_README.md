# Complete Progressive Tucker 133M → 257M project

This directory is a self-contained source-code snapshot. It already contains
the original training project plus the Progressive Tucker implementation; it
is **not** an overlay and does not require another repository checkout.

## Included

- all Python sources under `src/`;
- bundled third-party Python/CUDA sources under `third_party/`;
- all experiment launchers and the progressive launcher under `scripts/`;
- the complete test suite under `tests/`;
- `requirements.txt`, `environment.yml`, and `setup.sh`;
- the original `README.md` and the progressive design/run guide
  `PROGRESSIVE_TUCKER.md`.

Runtime data, model checkpoints, W&B state, logs, caches, secrets, and Git
history are intentionally not included. They are not source code and can be
large or machine/user-specific.

## Copy to another machine

Copy this whole directory with `rsync`, `scp`, an archive, or any file-sharing
service. No merge into another repository is needed.

## Environment

From this directory, either use the provided setup script:

```bash
bash setup.sh
```

or create the Conda environment and install the Python requirements manually:

```bash
conda env create -f environment.yml
conda activate huawei-stage2
python -m pip install -r requirements.txt
```

## Test

```bash
pytest -q tests/test_progressive_tucker.py \
  tests/test_tucker_lr_scaling.py \
  tests/test_tensorion.py
```

## Launch

Use a physical GPU UUID on a shared server:

```bash
export CUDA_VISIBLE_DEVICES=GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx
export WANDB_BASE_URL=https://wandb-radfan.ru
export WANDB_ENTITY=efficient-muon
export WANDB_MODE=online
bash scripts/launch_tucker133_to_257_progressive_firstorder.sh
```

The default effective batch is `32 * 4 = 128`. For less VRAM:

```bash
BATCH_SIZE=16 ACC_STEPS=8 EVAL_BATCH_SIZE=16 \
  bash scripts/launch_tucker133_to_257_progressive_firstorder.sh
```

See `PROGRESSIVE_TUCKER.md` for the rank-growth schedule, checkpoint/resume
behavior, memory requirement, and implementation details.
