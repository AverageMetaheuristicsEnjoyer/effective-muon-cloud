# Layerwise Tucker: depth, width and batch scaling

Follow-up to [the 257M optimization study](../layerwise-tucker-opt-20260917/README.md).
Branch: `codex/layerwise-tucker-opt-20260917`. This is a throughput/memory screen, not a training-quality study.

**Stopped after optimizer-scope clarification.** Both agent-created jobs were cancelled at 18:51:42 UTC on September 17 (Cloud cancellation acknowledged `status=deleted`, error code 0). The user requested comparisons including the earlier Riemannian update and dense controls. The AdamW-only matrix below is retained as the submitted protocol, not a completed experiment or the approved replacement optimizer matrix. Persistent NFS artifacts were not deleted; their contents have not yet been exported/verified.

## Registered matrix

Parameter counts below are obtained by constructing the actual models on the PyTorch meta device, including untied embeddings and LM head. These projections remain dense. Head dimension is always 128; FF width is always 2.75 times model width. Only layer count and width change.

| Profile | Layers | Width | Heads | FF width | Dense parameters | A quarter | B quarter | A half | B half |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| base | 12 | 1024 | 8 | 2816 | 257188864 | 169747696 | 142812460 | 255714544 | 201844012 |
| deep | 24 | 1024 | 8 | 2816 | 411354112 | 236471776 | 182601304 | 408405472 | 300664408 |
| wide | 12 | 2048 | 16 | 5632 | 822659072 | 472746224 | 365005100 | 816466160 | 600983852 |
| deep-wide | 24 | 2048 | 16 | 5632 | 1439270912 | 739445216 | 523962968 | 1426885088 | 995920472 |
| large | 32 | 2560 | 20 | 7040 | 2826734080 | 1368689792 | 919768352 | 2800782464 | 1902939424 |

Each profile runs dense, A quarter, B quarter, A half, B half at microbatch 1/4/16/32. All use reordered contractions and blockwise `torch.compile(max-autotune)`. Native kernels at every batch; matched Liger at microbatch 32. Total: 100 native + 25 Liger attempts. No manual Triton path or original eager baseline in this scaling screen: the previous study established the selected compiler policy.

## Controlled protocol

- Cloud.ru `torch28`, one exclusively allocated GPU per job; at most two jobs for this screen. Actual hardware and library versions are recorded by each result, rather than inferred from the resource label.
- Sequence length 1024, effective batch 32 sequences / 32768 tokens per optimizer step. Accumulation is 32/8/2/1 for microbatch 1/4/16/32. Unlike the previous 16384-token study, this accommodates microbatch 32. Do not directly compare the raw step milliseconds across the studies.
- BF16 autocast, FP32 parameters/residual stream/AdamW state, TF32 disabled, no activation checkpointing. Liger does not change residual precision. Persistent grad buffers when accumulation exceeds one; set-to-none otherwise.
- Direct tensorized initialization, no pretrained decomposition. Vocabulary 50304, same optimizer and clipping as the previous study. Synthetic fixed inputs and seed 42; loss only checks finite computation, not model quality.
- Five warmup steps, twenty timed full optimizer steps. Initialization/compilation/autotuning, input loading, evaluation and saving are excluded from steady-state time. Warmup is retained separately. Compiler caches are shared within each job group, so startup times are not cold-start comparisons.
- One run per matrix cell: preliminary screening, not three-seed confirmation. Dense/Tucker and native/Liger controls for a profile run sequentially on the same job/GPU. Small group: base/deep/wide (75 attempts); large group: deep-wide/large (50 attempts). Cross-group GPU differences must be checked before pooling curves.
- Record actual parameter counts, initialization time, full-step and phase timings, tokens/sec, allocated and reserved peaks. CUDA Graph private-pool memory caveats from the earlier report still apply; allocated bytes alone are not required VRAM.
- OOM is a capacity result, not a successful timing or a zero-throughput measurement. No automatic reduction of batch/precision or enabling checkpointing. Other errors/timeouts are failures. Each cell is a fresh process, timeout 1200 seconds. Manifests preserve every outcome; existing manifests/results are not overwritten on relaunch.

Before each group: CPU algebra/geometry tests and four GPU numerical gates covering both rank fractions, native/Liger, dense/A/B, three AdamW updates and accumulation 2. GPU gates use the group's maximum width (2048 or 2560), two layers and a small vocabulary/sequence; full-depth timing checks finite loss/gradients but is not a full-depth equivalence or convergence proof. Existing BF16 sanity tolerances are unchanged.

## Launch / persistence

Entrypoint `scripts/cloud_layerwise_tucker_opt.sh`, arguments `scaling small` or `scaling large`.
Use distinct `LAYERWISE_OPT_RESULTS` roots:

- `/workspace-SR006.nfs3/layerwise-tucker-scale-20260917/small`
- `/workspace-SR006.nfs3/layerwise-tucker-scale-20260917/large`

Each root retains `logs/`, correctness JSON, `scaling/manifest-<group>.json`, individual raw timing/OOM JSON, and `scaling.exit`. Platform completion is not sufficient: check application exit, all planned outcomes, correctness gates and persisted artifacts. `export scaling` and `export checks` in the existing launcher provide chunked SHA-256-protected export; `peek` reads persisted status without using a GPU.

Local checks before submission: five unit tests (including meta-device construction for all 30 profile/variant/rank combinations), an overridden-geometry CPU training smoke, dry-run matrix enumeration, shell syntax and diff checks. Submission IDs and live status are recorded in `jobs.json` after launch.

Both jobs were accepted and observed in platform `running` state on September 17 at approximately 18:50 UTC. Initial log retrieval timed out after 50 seconds; numerical gate completion and actual GPU identity have not yet been verified. Source revision submitted: `45057e782942ffd15df486ff37b1a313f21fff19`. No timing/quality results are claimed at this handoff.

Optimizer clarification: this series and the preceding optimization series use ordinary fused AdamW directly on Tucker factors/cores. Neither Muon, tangent-space projection nor Riemannian retraction is included. Full-step speedups must not be transferred to the earlier Riemannian training method without a separate optimizer comparison.

The earlier production late-growth launcher uses Tensorion core updates, Riemannian Muon factors, QR gauge fixing with momentum transport, `tucker_lr_scaling_mode=none` and no post-NS tangent projection. The older calibrated launcher differs in its LR scaling. Neither is a drop-in switch for the new layerwise model: attention has an unfactorized head axis, MLP factors are shared across roles, and the existing module traversal targets `TuckerLinear`. The old grouped QR implementation falls back to the sequential production path when vector transport is enabled; the old grouped Muon kernels implement vanilla MuonLite, not the Riemannian Tensorion update. Reuse requires an explicit adaptation and numerical parity tests, not simply enabling those flags.
