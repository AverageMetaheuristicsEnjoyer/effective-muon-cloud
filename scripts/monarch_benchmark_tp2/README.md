# Correct TP=2 Monarch-Muon benchmark

This is a two-GPU systems microbenchmark for Llama-like projection geometry.

- `dense_duplicated`, `dense_distributed`, and `dense_blockwise` use NVIDIA
  Megatron Core `ColumnParallelLinear` / `RowParallelLinear` and the official
  `TensorParallelMuon` implementation with its matching `muon_tp_mode`. The
  BF16 model weights use FP32 master copies for its Newton-Schulz step, as
  required by that optimizer implementation.
- `monarch_muon` uses the repository's batched `MonarchMuonOptimizer` and N=2
  rectangular Monarch factors. Each rank owns its local L/R factor blocks.

Every variant is a real LLaMA decoder: vocab-parallel embedding/head, RoPE,
causal `scaled_dot_product_attention`, RMSNorm, and SwiGLU. The `llama7b`
profile exactly matches the single-GPU benchmark's 6.89B dense LLaMA geometry:
32 layers, hidden size 4096, 32 heads, and FFN size 11008.

Monarch keeps hidden states sharded. A projection executes local BMM, an
all-to-all to change factor ownership, local BMM, then a second all-to-all to
return to canonical contiguous feature shards. The second collective is needed
for correct residuals, norms, and subsequent layers; it is intentionally timed.

The benchmark records CUDA-event time for forward, backward, optimizer, and
Newton-Schulz, plus API-level per-rank collective input bytes and peak allocated
GPU memory. Bytes are not fabric-wire bytes, which depend on NCCL's chosen
algorithm and topology. NCCL event time is a sum of collective event durations;
the end-to-end critical-path measure is `step_ms`.

## Setup

The H200 environment must contain the pinned official dependencies:

```bash
.venv/bin/pip install megatron-core==0.16.1
.venv/bin/pip install -e third_party/Emerging-Optimizers
```

## Run

Use an exclusive GPU pair. `CUDA_DEVICE_MAX_CONNECTIONS=1` enables the usual
Megatron ordering for asynchronous TP gradient reductions.

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
.venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m scripts.monarch_benchmark_tp2.benchmark_tp2 \
  --geometry small --variant dense_distributed \
  --warmup-steps 200 --measured-steps 500 \
  --output results/tp2_dense_distributed_small.json
```

For the first Monarch run, add `--validate-monarch`. It compares a distributed
rectangular N=2 projection against `blockdiag_butterfly_multiply` before timing.

To run all four variants on a known exclusive pair:

```bash
bash scripts/monarch_benchmark_tp2/run_all.sh 0,1 small
```

## Contended-GPU instrumentation smoke test

`smoke` retains the TP=2 model layouts, autograd-aware all-to-all calls, real
Muon/NS optimizer steps, and CUDA-event measurement, but uses `hidden=256`,
two layers, 16 tokens, and batch size one. It is suitable for checking that
measurements and JSON output work when only a small amount of memory is free;
do not use its latency results for a systems comparison while other jobs share
the GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1 CUDA_DEVICE_MAX_CONNECTIONS=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=2 \
  -m scripts.monarch_benchmark_tp2.benchmark_tp2 \
  --geometry smoke --variant monarch_muon --validate-monarch \
  --warmup-steps 3 --measured-steps 5 \
  --output results/monarch_tp2/smoke_monarch_muon.json
```

To check all four variants, run `bash scripts/monarch_benchmark_tp2/run_smoke.sh 0,1`.

## Systems kill-test grid

Use four LLaMA-like blocks first, with `TP=2`, microbatch one, and no gradient
accumulation. Sweep each width over local token counts 128, 512, and 2048;
repeat a selected point with `--layers 8` only after the four-layer results are
stable. The default run has 200 warmup and 500 measured steps.

```bash
bash scripts/monarch_benchmark_tp2/run_kill_test.sh 5,6 large 2048 4
```

The implemented widths are `small=1024`, `medium=2048`, and `large=4096`.
`--tokens` and `--layers` override a profile's default shape.

For the exact 6.89B single-GPU benchmark model, use a free pair and start
with 20 warmup plus 100 measured steps:

```bash
bash scripts/monarch_benchmark_tp2/run_kill_test.sh 5,6 llama7b 1024 32 20 100
```

For the full four-mode run with explicit memory admission, per-run config,
logs, phase timings, Newton-Schulz timings, and per-collective NCCL breakdown:

```bash
bash scripts/monarch_benchmark_tp2/run_llama7b_tp2_benchmark.sh 0,5 20 100 1024
```

It requires 65 GiB free on each GPU by default. Set `MIN_FREE_GIB` only when
you have independently confirmed a smaller stable margin. Optimizer hyperparameters
can be overridden with `LR`, `MOMENTUM`, `BETA1`, `BETA2`, and `WEIGHT_DECAY`.
