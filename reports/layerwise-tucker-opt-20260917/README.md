# Layerwise Tucker optimization experiments

Branch: `codex/layerwise-tucker-opt-20260917`. Experiments completed on Cloud.ru H100 80GB, PyTorch 2.8.0+cu128.

## Outcome: three-seed microbatch-16 confirmation

The fastest tested policy is **reordered contractions + blockwise torch.compile(max-autotune), without Liger**. When accumulating gradients, preallocate persistent grad buffers and preserve them across zeroing. The custom packed SwiGLU does not add a repeatable speed gain on top of the compiler. Liger is a memory/speed tradeoff, not the fastest fixed-microbatch policy here.

All 42 microbatch-16 confirmation runs and 18 microbatch-1 runs complete successfully. Per microbatch, all arms run sequentially on the same GPU/job. Seeds are 42/43/44; policy order is reversed for seed 43. A time is the median of three run medians; parentheses give min–max across those medians, not a confidence interval. Speedup is the median paired reference/optimized ratio, **relative to that arm's original implementation, not to dense**. Opt/dense is normalized to the equally compiled dense control.

| Arm | Reference ms (range) | Optimized ms (range) | Compile + custom pointwise ms | Paired speedup | Opt/dense time |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 120.42 (120.41–120.82) | 62.42 (62.41–62.44) | — | 1.93x | 1.000x |
| A 0.25 | 118.97 (118.52–119.36) | 52.51 (52.23–52.53) | 52.57 | 2.26x | 0.841x |
| B 0.25 | 119.31 (119.27–119.36) | 51.59 (51.58–51.59) | 51.84 | 2.31x | 0.827x |
| A 0.5 | 135.55 (135.40–136.02) | 60.69 (60.62–60.70) | 60.82 | 2.24x | 0.972x |
| B 0.5 | 134.98 (134.97–135.21) | 59.52 (59.46–59.56) | 59.76 | 2.27x | 0.954x |

Quarter-rank Tucker needs about 16–17% less step time than equally optimized dense; half-rank variants are close to dense. The extra manual pointwise kernel has paired median time ratios 1.001/1.005/1.002/1.004 for A-quarter/B-quarter/A-half/B-half, respectively: no additional speed win. These measurements concern throughput, not equal model quality.

| Arm | Reserved GB: reference / optimized | Optimized initialization ms | Optimized five-step warmup seconds, range |
| --- | ---: | ---: | ---: |
| dense | 26.29 / 22.25 | 107.6 | 7.70–8.74 |
| A 0.25 | 27.13 / 22.02 | 231.3 | 13.18–75.29 |
| B 0.25 | 26.90 / 23.42 | 217.8 | 10.94–12.18 |
| A 0.5 | 29.19 / 25.00 | 195.0 | 11.64–13.42 |
| B 0.5 | 28.85 / 24.66 | 229.1 | 11.60–73.89 |

Initialization is model construction and direct factor initialization after CUDA context initialization, not pretrained-weight decomposition. Warmup includes compilation, autotuning, graph capture and five training steps. Inductor/Triton caches are shared across these jobs, so this is **not** a standardized cold-start benchmark. Initialization and warmup are excluded from the reported step times. The memory caveat below explains why allocated bytes must not be advertised as required VRAM.

## Protocol

H100 80GB on Cloud.ru, image `torch28`, PyTorch 2.8, sequence length 1024, 16,384 tokens per optimizer step. Microbatch 1 means accumulation 16; microbatch 16 means accumulation 1. Five warmup optimizer steps, then twenty timed steps. Timing includes forward, backward, gradient clipping, fused AdamW, zeroing and synchronization; excludes initialization, warmup/compilation, input loading, evaluation and checkpointing. Synthetic fixed tokens are reused: losses are correctness/finite-value diagnostics, not model-quality measurements.

BF16 autocast, FP32 parameters, optimizer state and residual stream, TF32 disabled, no activation checkpointing. In Liger arms, Liger 0.8.1 is enabled equally for dense and Tucker, including RMSNorm, SwiGLU and fused linear cross entropy. The benchmark explicitly disables the existing Liger-only BF16 residual cast to avoid changing precision between arms.

The model has 12 layers, width 1024, FF width 2816, 8 heads of size 128 and vocabulary 50,304. A shares gate/up Tucker factors but has an independent down factorization; B also shares factors with transposed down. Rank fractions apply to spatial axes; role ranks remain full. Parameter counts: dense 257,188,864; A quarter/half 169,747,696 / 255,714,544; B quarter/half 142,812,460 / 201,844,012. This is direct tensorized initialization per the colleague specification, not decomposition of a pretrained checkpoint.

`reference` is the original contraction order. `reordered` first contracts the FF factor with the small role cores, then projects token latents with the resulting partial weights; it does not materialize the full dense MLP weight. Parameters, ranks, initialization and optimizer are unchanged.

Each initial screen below uses seed 42. Compare within-stage controls first; separate jobs can differ in CPU scheduling and GPU conditions. CPU affinity and raw samples are retained. The initial screen's peak memory column is allocated bytes in decimal GB, not reserved VRAM or activation-only memory; other tables label the counter explicitly. Profiling runs occur after timing and memory collection.

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

This is roughly a 5–6% full-step gain, not the near-2x gain of the isolated suboperation. Neither custom path beats eager reordered at microbatch 1 in this screen. Compiler fusion is a much larger full-model opportunity; the combination and confirmation stages measure additive gains separately.

The first compile+Liger combination failed to trace a bound `autograd.Function.apply` saved on the Tucker module. The fixed integration keeps the Function class visible to Dynamo; custom Triton entrypoints similarly call `.apply` from ordinary functions. The failed job log is retained in `logs/combined-bound-apply-failure.txt`.

## Combined kernels: memory versus speed

All entries below use blockwise max-autotune and microbatch 16. Native means native RMSNorm/SwiGLU/CE inside and outside the compiler; the model still uses reordered Tucker contractions.

| Arm | Compile ms | + Liger ms | + Liger + custom pointwise ms | Allocated GB: compile / Liger | Reserved GB: compile / Liger |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 62.32 | 72.22 | — | 11.64 / 3.85 | 22.27 / 12.15 |
| A 0.5 | 60.55 | 73.61 | 69.21 | 11.61 / 3.82 | 24.95 / 14.94 |
| B 0.25 | 51.60 | 64.32 | 60.27 | 10.23 / 2.41 | 23.42 / 12.37 |

Liger reduces reserved memory substantially but makes these compiled steps slower. At microbatch 1, matched compile-only versus compile+Liger times are dense 144.19 / 345.07 ms, A 0.5 177.59 / 383.41 ms, B 0.25 140.09 / 342.86 ms. Liger moves some gradient computation into forward; individual forward/backward durations are not interchangeable with native CE, so use the full-step metric.

**CUDA Graph memory caveat:** replay does not repeat allocator calls, while graph-private pools must remain reserved. Consequently, low post-warmup `max_memory_allocated` values are not a capacity bound or the true maximum live intermediate storage during replay. For example, compiled B quarter + Liger reports 2.41 GB allocated but 12.37 GB reserved. Reserved includes allocator caching and is not a precise minimum either; neither number includes every non-PyTorch CUDA allocation. Both counters are retained, and no claim is made that this model fits in 2.41 GB VRAM. See [NVIDIA graph memory management](https://docs.nvidia.com/dl-cuda-graph/troubleshooting/memory-issues.html) and [graph replay / allocator behavior](https://docs.nvidia.com/dl-cuda-graph/torch-cuda-graph/torch-integration.html).

## Three-seed microbatch-1 confirmation

Same GPU/job, seed 42/43/44, paired original versus reordered + max-autotune. Each number is the median of three run medians; parentheses give their minimum–maximum, not a confidence interval. Speedup is the median of the three paired reference/optimized ratios. Policy order reverses for seed 43. The optimized path uses persistent gradient buffers; the original path retains its usual set-to-none behavior. Both paths preserve the same mathematical AdamW update and effective batch.

| Arm | Reference ms (range) | Optimized ms (range) | Paired speedup | Reserved GB: reference / optimized |
| --- | ---: | ---: | ---: | ---: |
| dense | 425.50 (424.91–425.87) | 144.38 (144.11–144.58) | 2.95x | 6.12 / 5.98 |
| A 0.5 | 621.03 (618.26–627.14) | 176.90 (176.83–177.58) | 3.51x | 6.54 / 6.26 |
| B 0.25 | 612.01 (611.88–612.75) | 140.54 (140.33–140.79) | 4.36x | 4.06 / 4.04 |

All eighteen measurements and both rank-fraction validation gates complete successfully. CUDA Graph compatibility was also exercised with 12 layers and gradient accumulation in the separate correctness tests. Raw evidence is under `confirm-mb1/`; job `lm-mpi-job-f7d26d9b-885d-455b-8c6a-b5d75b91f9a4`.

## Numerical validation and limits

Four CPU unit tests pass locally and in the Cloud bootstrap, including attention/MLP outputs and gradients, causality/layer independence, and three FP64 AdamW updates comparing the reference and reordered MLP. The reordered test uses output tolerance 1e-9 and gradient/parameter tolerance 1e-8.

The final GPU gates cover both rank fractions, dense/A/B, 12 layers, three AdamW updates, and accumulation 1 or 2 as appropriate. Full-size timing runs also check finite loss and gradient norm on every warmup/measured step. These are sanity gates, not a convergence proof: configured bounds are relative loss error <1%, worst per-parameter-tensor gradient error <10%, and worst parameter-tensor change relative to reference <2%.

The extra quarter-rank audit measures the selected reordered + max-autotune path against eager reference. Relative gradient error is L2(delta)/L2(reference). Global error concatenates all parameter gradients; worst-tensor error concerns a whole parameter tensor, not a scalar entry.

| Arm | Global grad error, identical initial weights | Global grad error after two prior updates | Maximum tensor-grad error over three steps | Maximum loss error |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.540% | 1.596% | 2.538% | 0.0323% |
| A quarter | 0.934% | 1.013% | 3.038% | 0.0206% |
| B quarter | 0.782% | 1.118% | 7.442% | 0.0301% |

The selected B-quarter path reaches 7.44% worst-tensor error in `transformer.h.1.attn.qkvo.role` at step 1. The manual-pointwise candidate reaches 9.06% in `transformer.h.11.mlp.role` at step 2, with global error 1.31%. Later-step comparisons include drift from earlier optimizer updates, so they do not isolate a single kernel's backward error. Initial identical-weight errors and the isolated Triton test are separate evidence. In the selected audit, the largest parameter-tensor relative difference after the three updates is 0.3964%.

Do not interpret “passed” as bitwise BF16 equivalence or equal final training quality. FP64 algebraic equivalence is tested; BF16 reassociation/compiler fusion changes rounding. Real-data pretraining/validation quality, long-run stability, and other optimizers remain untested.

## Reproduction

From this checkout with PyTorch 2.8/CUDA and the benchmark dependencies installed:

```bash
PYTHONPATH=src python scripts/benchmark_layerwise_tucker.py --variant B --rank-fraction 0.25 --microbatch 16 --execution reordered --compile-mode max-autotune --output result-mb16.json
PYTHONPATH=src python scripts/benchmark_layerwise_tucker.py --variant B --rank-fraction 0.25 --microbatch 1 --execution reordered --compile-mode max-autotune --stable-grad-buffers --output result-mb1.json
```

Cloud entrypoint: `scripts/cloud_layerwise_tucker_opt.sh`, image `torch28`, one GPU, `--no-pip`. Modes `confirm-mb16 native` and `confirm-mb1 native` reproduce the matrices. Use a fresh `LAYERWISE_OPT_RESULTS` directory to preserve this run's artifacts; remove/replace neither existing results nor another user's work. Compiling only the blocks, marking each microbatch with `torch.compiler.cudagraph_mark_step_begin()`, retaining loss scalars outside graph storage, and persistent `.grad` buffers for accumulation are part of the measured implementation. Production training defaults are not switched automatically.

## Evidence locations

Raw timing JSON: `screen/`, `liger/`, `compile/`, `compile-stable/`, `triton/`, `combined/`, `combined-mb1/`, `confirm-mb16/`, `confirm-mb1/`. Numerical results and the isolated Triton microbenchmark are under `checks/`. GPU profiles are inside the three profiled screen JSON files. Persistent Cloud root: `/workspace-SR006.nfs3/layerwise-tucker-opt-20260917`; application logs are under `logs/`.

Screen job: `lm-mpi-job-e254a568-9b7f-4cd3-8e9c-cd09ec5bd3ed`; Liger job: `lm-mpi-job-c39dc8ba-70c5-41b5-a09e-20b362d979ad`. Initial compile job: `lm-mpi-job-88d05ccc-0eaa-4b72-ba57-ff0f01a1e408`. Custom Triton screen: `lm-mpi-job-3b7ac38b-5c27-4331-927f-28a76a8c07d0`.

The complete GPU job/source revision inventory is in [jobs.json](jobs.json). This report retains 138 full-size timing attempts: 132 complete and six initial CUDA Graph accumulation failures; all 60 final confirmation measurements complete. There are also 19 complete numerical-validation files and one isolated Triton microbenchmark file.

The 158 raw JSON artifacts listed in [sha256.json](sha256.json) were exported in gzip/base64 chunks and checked against SHA-256 of the original NFS bytes. The manifest and job inventory themselves are local report metadata. A platform `Completed` state does not establish successful benchmarks; each final stage was checked against its JSON statuses and persisted application exit 0. The initial `compile.exit=1` documents the six fixed failures. An administrative CPU helper was once mistyped as `checks` rather than `export checks` and exited 127 at `nvidia-smi`; this is not a numerical-validation result and is excluded from GPU counts.
