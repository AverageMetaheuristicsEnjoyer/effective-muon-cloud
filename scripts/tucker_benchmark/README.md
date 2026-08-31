# Tucker parallel rank-8 scale benchmark

This harness compares four variants at the five established Llama scales:
Dense AdamW, Dense Muon, the archive's reference Tucker path, and its new
custom-backward + parallel-Muon + grouped-retraction path. The internal Tucker
linears use per-layer rank plans that keep the whole model within 0.03% of its
dense counterpart. Every selected rank and every mode is divisible by 8; the
hidden dimension and dense `lm_head` are unchanged. For the 257M iso-parameter
Tucker variants, the FFN width is 3072 instead of the dense baseline's 2816;
this keeps the natural 32x32 and 48x64 mode pairs on the fused path while all
ranks remain divisible by 8.

The common contract is sequence length 1024, 16,384 tokens per optimizer step,
microbatches 1/2/4/8/16, FP32 master parameters and optimizer state, BF16
autocast, Liger fused cross-entropy, and no activation checkpointing. Tucker
uses `chunked_contract`, the cross-scale-safe `recast` cache policy, Muon with
six Newton-Schulz steps, QR retraction every step, and no vector transport.

Each result records Forward, Backward, Forward+Backward, gradient clipping,
Optimizer, Retraction, and full-step time separately. It also records both the
Forward+Backward peak and the full-step allocated-memory peak. Random inputs are
preallocated; data loading, evaluation, checkpointing, and W&B are excluded.

```bash
python -m scripts.tucker_benchmark.run_sweep --exclusive-gpu
```

## Cloud.ru

Use the included entry point for the CPU contract test, CUDA correctness suite,
H100 stream tuning, and final one-GPU sweep:

```bash
mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image torch28 --no-pip \
  --gpus cpu --args selftest

mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image torch28 --no-pip \
  --gpus 1 --args correctness

mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image torch28 --no-pip \
  --gpus 1 --args autotune
```

Results persist under
`/workspace-SR006.nfs3/tucker-parallel-rank8-scale-20260826`.

The `progressive-ranks` Cloud command benchmarks the four lower 257M
progressive stages (133M, 160M, 190M, and 225M) both with their exact rank
plans and with parameter-matched rank-8 plans. It runs the reference and new
parallel Tucker paths at microbatches 1/2/4/8/16.

The `order3-screen` command compares the current order-4 layout with two
order-3 layouts at matched 133M and 225M model budgets. `order3_input` splits
the input dimension and leaves the output dimension intact; `order3_output`
does the reverse. The implementation keeps the existing four-mode kernels by
using one fixed singleton buffer, so each order-3 layer has exactly three
trainable factor matrices. The screen uses microbatches 1 and 16.

```bash
mlsub run --repo <public-mirror> --branch <branch> \
  --entry scripts/tucker_benchmark/cloud_run.sh --image torch28 --no-pip \
  --gpus 1 --args order3-screen
```
