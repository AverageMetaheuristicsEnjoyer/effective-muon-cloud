# Tucker rank-8 scale benchmark

This benchmark measures dense AdamW, dense Muon and static Tucker at the five
Llama scales from `monarch_benchmark`: 257M, 834M, 1.44B, 3.48B and 6.89B.
Every Tucker mode and rank is divisible by 8. Per-layer ranks are selected so
the total Tucker parameter count stays within 0.03% of the corresponding dense
model without widening `hidden_dim`. Every Tucker layer is forced through
factorized contractions; materializing an effective dense weight is forbidden.

All variants at a scale use the same Llama geometry and preallocated random
tokens. Data loading, evaluation, checkpointing and W&B are excluded. A sample
contains gradient accumulation, forward, backward, gradient clipping, optimizer
step and zeroing. The Tucker optimizer slice also includes QR retraction and
optimizer-state vector transport. Forward, backward and their sum are reported
separately from the optimizer. The Tucker core and four factor directions are
computed on five CUDA streams before the coupled LR scaling.

The default sweep matches the existing `monarch_benchmark` scale and capacity protocol:
sequence length 1024, 16,384 tokens per optimizer step, microbatch 1/2/4/8/16,
three warmup steps and twelve measured steps. Parameters and optimizer state are
BF16; the Tucker spectral power-iteration vectors remain FP32 as in training.

```bash
python -m scripts.tucker_benchmark.run_sweep --exclusive-gpu
```

The production batch/accumulation point from the launcher is microbatch 32 x 4:

```bash
python -m scripts.tucker_benchmark.run_sweep \
  --output-dir benchmark_results/tucker-rank8-production \
  --models 257m --microbatches 32 --tokens-per-step 131072
```

## Cloud.ru

The entry point expects the repository on a public HTTPS git branch. Run the CPU
rehearsal first, then the one-H100 sweep:

```bash
ssh brain_lab mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image te2 --no-pip \
  --gpus cpu --args selftest

ssh brain_lab mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image te2 --no-pip \
  --gpus 1
```

Results persist under `/workspace-SR006.nfs3/tucker-scale-rank8-membench-20260826`. Use `peek` while
the sweep is running and `export` afterwards; both are CPU jobs using the same
entry point.
