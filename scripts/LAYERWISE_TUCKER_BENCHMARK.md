# Layerwise Tucker A/B timing

Implements the supplied `TUCKER_MODEL_TENSORIZATION.md` equations: one Q/K/V/O
bank per MHA layer, separate Q/O and K/V banks for GQA, unfactorized head axis,
dense cores, MLP A (joint gate/up and independent two-mode down) and B (joint
gate/up/down transpose). Each layer owns its parameters. Full projection
matrices are never constructed in the Tucker model or its forward. Norms,
RoPE, causal SDPA, residuals, embeddings and dense LM head are preserved;
SwiGLU operates at the original FFN width.

This is direct initialization and training of a factorized model, not an
SVD/HOOI conversion of a pretrained checkpoint. Initialization uses Gaussian
hidden/FFN/head factors normalized by rank and orthogonal role factors;
core variance matches the intended effective weight variance in expectation.
Residual O/down roles use the baseline depth scaling. No QR retraction or
special Tucker optimizer is assumed by the specification.

## Experiment

- Baseline: 12 layers, hidden 1024, 8 heads of dimension 128, FFN 2816,
  vocabulary 50304; 257,188,864 dense parameters.
- Dense, A at rank fraction 0.25/0.5, B at rank fraction 0.25/0.5.
- Attention ranks `(256,32,4)` or `(512,64,4)`; MLP ranks
  `(704,256,2/3)` or `(1408,512,2/3)`. A down ranks match `(rm,rff)`.
- MHA is benchmarked; MHA and GQA both receive numerical correctness tests.
- BF16 autocast, FP32 parameters/state, PyTorch causal SDPA, native RMSNorm,
  native SwiGLU and cross entropy for all arms. No compile or checkpointing.
- AdamW, LR 1e-4, weight decay 0.01, clipping at 1.0; no training-quality claim.
- Sequence 1024; 16,384 tokens/optimizer step. Microbatch 1/16 with accumulation
  16/1. Identical preallocated synthetic inputs and labels in each process.
- Three processes per point (seeds 42/43/44), rotating arm order; 5 warmup and
  20 measured steps each: 30 independent benchmark processes.
- CUDA-event forward/backward/clipping/optimizer/zero-grad timings and
  synchronized host step timing. Initialization excludes imports, CUDA context,
  optimizer and data allocation. Record peak allocated/reserved memory after
  warmup, initialization memory separately, actual parameter shapes/counts,
  loss, versions, GPU identity and Git revision.

These runs compare equal functional dimensions, not equal parameter counts or
equal quality. Initialization is one sample per process (three per point),
so it is a coarse engineering measurement. No data loading, validation,
checkpoint saving or network logging is inside the timing window.

## Run

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_layerwise_tucker.py -v
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/effective-muon-cloud.git \
  --branch codex/layerwise-tucker-timing-20260917 \
  --entry scripts/cloud_layerwise_tucker.sh --image torch28 --no-pip \
  --gpus cpu --args selftest
# After APPLICATION_EXIT=0:
mlsub run --repo https://github.com/AverageMetaheuristicsEnjoyer/effective-muon-cloud.git \
  --branch codex/layerwise-tucker-timing-20260917 \
  --entry scripts/cloud_layerwise_tucker.sh --image torch28 --no-pip \
  --gpus 1 --args sweep
```

Logs, per-step samples and JSON results persist under
`/workspace-SR006.nfs3/layerwise-tucker-20260917`. `export` returns complete JSON
artifacts through the gateway logs. Platform `Completed` alone does not prove
success: check `APPLICATION_EXIT=0` and all 30 result files.
