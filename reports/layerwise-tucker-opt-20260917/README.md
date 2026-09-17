# Layerwise Tucker optimization experiments

Branch: `codex/layerwise-tucker-opt-20260917`. Work in progress: final combined-kernel validation and repeated comparisons are not yet complete.

## Protocol

H100 80GB on Cloud.ru, image `torch28`, PyTorch 2.8, sequence length 1024, 16,384 tokens per optimizer step. Microbatch 1 means accumulation 16; microbatch 16 means accumulation 1. Five warmup optimizer steps, then twenty timed steps. Timing includes forward, backward, gradient clipping, fused AdamW, zeroing and synchronization; excludes initialization, warmup/compilation, input loading, evaluation and checkpointing. Synthetic fixed tokens are reused: losses are correctness/finite-value diagnostics, not model-quality measurements.

BF16 autocast, FP32 parameters, optimizer state and residual stream, TF32 disabled, no activation checkpointing. Liger 0.8.1 is enabled for both dense and Tucker, including RMSNorm, SwiGLU and fused linear cross entropy. The benchmark explicitly disables the existing Liger-only BF16 residual cast to avoid changing precision between arms.

`reference` is the original contraction order. `reordered` first contracts the FF factor with the small role cores, then projects token latents with the resulting partial weights; it does not materialize the full dense MLP weight. Parameters, ranks, initialization and optimizer are unchanged.

Each initial screen below uses seed 42. Compare within-stage controls first; separate jobs can differ in CPU scheduling and GPU conditions. CPU affinity and raw samples are retained. Peak memory is allocated bytes in decimal GB, not reserved VRAM or activation-only memory. Profiling runs occur after timing and memory collection.

## Initial microbatch-16 screen

| Arm | Reference ms | Reordered ms | Reordered + Liger ms | Peak GB: reference / reordered / Liger |
| --- | ---: | ---: | ---: | ---: |
| dense | 119.99 | 119.99 | 105.61 | 23.30 / 23.30 / 12.79 |
| A 0.25 | 118.46 | 112.70 | 97.97 | 23.30 / 22.51 / 12.01 |
| B 0.25 | 117.72 | 112.21 | 97.25 | 22.93 / 22.15 / 11.64 |
| A 0.5 | 135.34 | 121.73 | 107.22 | 26.21 / 24.67 / 14.18 |
| B 0.5 | 135.12 | 121.36 | 107.18 | 25.46 / 23.90 / 13.42 |

The plain reordered implementation improves this full-step screen by about 5% at quarter ranks and 10% at half ranks. The theoretical MLP FLOP ratio is not a full-model speed prediction. Eager microbatch-1 measurements show no corresponding improvement; they remain sensitive to host launch overhead.

## Compiler and custom Triton screens

Compilation is blockwise, full-graph, static shape. Embedding, final norm, LM head, cross entropy and optimizer remain outside the compiled blocks. Both compiler modes use CUDA Graphs in these tests; `max-autotune` additionally tunes matrix kernels. Initial microbatch-16 measurements:

| Arm | reduce-overhead ms | max-autotune ms |
| --- | ---: | ---: |
| dense | 62.01 | 61.66 |
| A 0.5 | 61.75 | 59.99 |
| B 0.25 | 52.37 | 51.04 |

All six initial microbatch-1 compiler cases failed during backward with a CUDA Graph storage-overwrite error. These failures are retained in `compile/`, not treated as performance results. The two-layer, accumulation-1 test was insufficient. The strengthened test uses 12 layers, accumulation 2 and three AdamW updates.

Preallocating zeroed `.grad` buffers before compiled backward and preserving them with `zero_grad(set_to_none=False)` resolves accumulation: all twelve `compile-stable/` cases complete. With max-autotune, microbatch-1 step times are dense 148.09 ms, A 0.5 180.90 ms, B 0.25 144.83 ms. Preallocation increases memory and adds zeroing/copying work, so use it when accumulating, not as a free optimization for accumulation 1. The [CUDA Graph lifetime documentation](https://docs.pytorch.org/docs/main/user_guide/torch_compiler/torch.compiler_cudagraph_trees.html) describes replay storage lifetimes; the local PyTorch 2.14 source also explicitly recommends persistent grad buffers for this error. Actual cloud tests here stay on PyTorch 2.8.

Two custom Triton paths have output/gradient checks: packed gate/up SwiGLU with fused pointwise backward, and fused gate/up GEMMs plus SwiGLU with PyTorch GEMMs for input/weight gradients. Microbenchmark outputs match native BF16 outputs exactly for the four tested shapes; relative gradient errors are at most 1.69e-5. At 16,384 tokens the packed path takes 0.592/0.711 ms for ranks 256/512 versus native 1.118/1.205 ms; the fused-GEMM path takes 0.644/0.787 ms. At 1024 tokens both custom paths are slower.

Full-step microbatch-16 comparison on the same Triton-screen job:

| Arm | Reordered ms | Packed pointwise ms | Fused GEMM + SwiGLU ms |
| --- | ---: | ---: | ---: |
| A 0.5 | 122.18 | 115.32 | 115.73 |
| B 0.25 | 111.63 | 105.77 | 105.23 |

This is roughly a 5–6% full-step gain, not the near-2x gain of the isolated suboperation. Neither custom path beats eager reordered at microbatch 1 in this screen. Compiler fusion is a much larger full-model opportunity; additive gains from combining these approaches require their own measurements.

The first compile+Liger combination failed to trace a bound `autograd.Function.apply` saved on the Tucker module. The fixed integration keeps the Function class visible to Dynamo; custom Triton entrypoints similarly call `.apply` from ordinary functions. The failed job log is retained in `logs/combined-bound-apply-failure.txt`.

## Evidence locations

Raw JSON: `screen/`, `liger/`, `compile/`. GPU profiles are inside the three profiled screen JSON files. Persistent Cloud root: `/workspace-SR006.nfs3/layerwise-tucker-opt-20260917`; application logs are under `logs/`.

Screen job: `lm-mpi-job-e254a568-9b7f-4cd3-8e9c-cd09ec5bd3ed`; Liger job: `lm-mpi-job-c39dc8ba-70c5-41b5-a09e-20b362d979ad`. Initial compile job: `lm-mpi-job-88d05ccc-0eaa-4b72-ba57-ff0f01a1e408`. Custom Triton screen: `lm-mpi-job-3b7ac38b-5c27-4331-927f-28a76a8c07d0`.

All retrieved JSON files are exported in gzip/base64 chunks, decoded with a SHA-256 check against the original NFS bytes. A platform `Completed` state does not establish successful benchmarks; check each JSON status and the persisted application exit.
