# Dense Muon versus layerwise Riemannian Tucker

**Final scaling result:** all 150 configurations processed: 123 complete timing runs and 27 CUDA OOM outcomes, with no other failures after storage recovery. B-quarter is 1.24–2.74x faster than the faster matched dense Muon control at every comparable completed point. A-quarter is 1.06–1.89x faster; both half-rank variants remain slower. Full tables, absolute times, parameter counts and limitations: [SCALING_RESULTS.md](SCALING_RESULTS.md). Raw CSV: [scaling-results.csv](scaling-results.csv). H100, 32768 tokens/step, TF32 off, one screening seed; no equal-quality claim. The pilot below used only 16384 tokens/step and must not be pooled with scaling times.

Requested primary comparison: dense + production vanilla Muon versus Tucker A/B + the prior Tensorion/Riemannian-Muon update, including QR retraction and differential momentum transport in every timed optimizer step. Not a comparison against the original eager Tucker implementation or an AdamW baseline. The earlier AdamW-only scaling jobs were cancelled; see the [submission audit](../layerwise-tucker-scale-20260917/README.md).

## Update semantics

Dense uses `third_party/lite/muonlite.py:MuonLite` with LITE disabled: LR 1e-3, weight decay 0.1, momentum 0.95, six production Newton-Schulz iterations; embeddings/head/norm use its unchanged AdamW fallback (betas 0.9/0.99). Grouped dense is a scheduling-only candidate: same coefficients, momentum and fallback, batched equal-shaped matrices, original compiled NS kernel. Both original and grouped controls must be measured before selecting a baseline.

Tucker uses the last late-growth launcher's update semantics: Tensorion heavy-ball cores, Riemannian Muon factor gradients projected onto their Stiefel tangent spaces, six quintic NS iterations, LR 1e-3, weight decay 0.1, momentum 0.95, no coupled LR calibration, no post-NS direction projection. Non-tensorized matrices use AdamW (0.9/0.99); norms have zero weight decay as in the Tensorion parameter groups. This is an adaptation of that update rule to the new parameterization, not a claim that it is the old four-mode model. Dense MuonLite and Tensorion intentionally retain their respective production NS polynomials and momentum formulas; they are not silently substituted for one another.

Before optimization, QR-gauge-fix the initialized factors and absorb R into the cores, preserving the effective initialized tensors. Attention: retract only hidden/head-dimension/role factors, not the uncontracted head-index axis. MLP A: shared gate/up factors plus independent two-mode down; B: shared three-role factors, each updated/retracted once. Differential QR transport updates both factor momentum and the core contribution from dR. The reference reuses the original QR/transport functions; the grouped candidate batches independent equal-shaped modules without dropping transport. Initialization includes the initial gauge fix and is outside steady-state timing.

All models: BF16 forward/backward autocast, FP32 parameters/residuals/optimizer state, TF32 disabled, no checkpointing, unchanged synthetic input and clipping protocol. NS, QR and transport preserve FP32 precision; no undocumented BF16 optimizer shortcut. Reordered contractions + blockwise max-autotune are used equally. Full step includes forward/backward, accumulation, clipping, optimizer, retraction/transport, zeroing and synchronization. Compilation/warmup excluded. Separate CUDA event metrics record optimizer and retraction/transport, with synchronized host full-step time as the primary metric.

## Validation and launch sequence

1. CPU FP64 tests: initial effective-tensor preservation (MHA/GQA, A/B), five grouped-versus-sequential updates and momentum states, factor tangency, finite-difference differential transport, and grouped dense parity with production MuonLite. Together with existing geometry/algebra tests: nine tests.
2. GPU gate at width 1024, both ranks, dense/A/B, three updates. Identical gradients from a BF16 reference backward isolate optimizer/transport error from two diverging backward trajectories. Check parameter/momentum tensor-relative errors <0.003 and BF16 logits relative error <0.01; these are implementation gates, not quality/convergence claims.
3. Eight full-size pilot runs at base geometry, microbatch 16, sequence 1024, 16384 tokens/step, five warmup and ten measured steps: dense original/grouped; A-half and B-quarter original/grouped-optimizer/grouped-optimizer-and-QR. New grouped kernels are candidates until measured, not assumed faster.
4. Use verified policies in the depth/width/batch screen: dense 257M/411M/823M/1.439B/2.827B; Tucker A/B at quarter/half ranks; microbatch 1/4/16/32 and fixed 32768 tokens/step. Persistent grad buffers when accumulating. Matched Liger at the largest batch is a separate memory/speed comparison. Capacity failures must remain explicit; no silent checkpointing or batch change.

Cloud entrypoint: `scripts/cloud_layerwise_tucker_opt.sh riemann-pilot`, image `torch28`, one GPU. Fresh persistent root `/workspace-SR006.nfs3/layerwise-tucker-riemann-20260917/pilot`. Source/job IDs and measured outcomes are recorded after submission. No performance or quality result is claimed by this preregistered protocol.

## First strict-FP32 pilot: completed

H100 80GB HBM3, all eight timing runs and both parity files complete; application exit 0. One seed, 5 warmup + 10 measured steps, 16384 tokens/step, no Liger. These are screening measurements, not repeated confirmations.

| Arm | Original optimizer + QR, ms | Grouped optimizer only, ms | Grouped optimizer + QR, ms |
| --- | ---: | ---: | ---: |
| Dense Muon | 225.40 | 195.31 | — |
| A half | 706.46 | 663.47 | 660.78 |
| B quarter | 231.47 | 205.90 | 200.69 |

Neither Tucker arm yet beats the equally optimized dense control. A-half spends about 410 ms in QR/transport and 194 ms in the grouped optimizer; B-quarter spends about 115 ms and 36 ms respectively. Grouping NS helps, but grouping the original QR alone barely addresses the dominant cost.

GPU parity (both fractions, 3 identical-gradient updates): maximum parameter-tensor relative error 2.03e-6, momentum-tensor relative error 2.81e-6. BF16 logits relative error reaches 0.00593; this is not bitwise equivalence or a convergence claim. Raw parity JSON was exported with verified SHA-256.

## Follow-up candidates

- Production `src/main.py` enables TF32, whereas the preceding AdamW and first Riemannian pilot prohibited it. A separate matched TF32 pilot is therefore registered; do not pool its measurements with strict-FP32 timings. QR orthogonality/tangency diagnostics are evaluated with TF32 disabled, outside timing. A TF32 optimization is a precision-policy comparison, not scheduling-only equivalence.
- Retraction candidate: Cholesky QR after an update from an orthonormal factor. Compute the Gram in IEEE FP32, obtain positive-diagonal R from its Cholesky factor, then Q by triangular solve; transport uses the same differential QR equations. Initialization still uses Householder QR. In exact arithmetic this is the same positive-diagonal thin QR map for full-rank factors, but squared conditioning requires numerical checks. Five CPU tests in the optimizer suite pass, including five FP64 updates/state comparisons against the original QR. GPU parity and same-job dense/grouped-QR controls are required before selecting it. No fallback or speed claim is assumed.
- The scaling screen adds both original and grouped dense controls at every point, yielding 150 attempts (90 small-group, 60 large-group). The speedup denominator must use the faster successful dense implementation at the same geometry, batch and Liger policy, not the slower original dense. All scaling runs use 32768 tokens/step and must not be mixed with this 16384-token pilot.

**TF32 pilot rejected by the numerical gate.** At the first A-quarter update, maximum momentum-tensor relative error was 0.0151046 (limit 0.003), parameter error 7.18e-5, BF16 logits error 0.006605. Application exit 1; no timing matrix ran. The failing parameter/stage was not isolated by this gate, so this does not establish that QR alone caused the error. Keep strict-FP32 comparisons separate; do not loosen the threshold or claim production-TF32 speedups from the strict results. The failure log is retained under `logs/`.

## Selected Cholesky policy: same-job controls

All five runs and both rank-fraction GPU gates complete with application exit 0. Same precision, batch, warmup and sample counts as the first pilot; optimizer grouping is enabled in every row below. A QR timing includes differential momentum transport. Dense has no retraction.

| Arm | Householder QR full step, ms | Cholesky QR full step, ms | QR/transport: Householder -> Cholesky, ms | Selected time / dense |
| --- | ---: | ---: | ---: | ---: |
| Dense Muon | 193.61 | — | — | 1.000 |
| A half | 660.02 | 368.92 | 410.57 -> 117.33 | 1.906 |
| B quarter | 200.25 | 109.78 | 114.96 -> 24.50 | 0.567 |

For B-quarter, this saves about 43.3% of full-step time relative to the same-job optimized dense and about 45.2% relative to its own grouped Householder implementation. A-half's own retraction optimization saves about 44.1% of full-step time but is not a win over dense. Parameter counts are dense 257188864, A-half 255714544, B-quarter 142812460: equal architecture dimensions, not equal parameter count or quality.

Cholesky GPU parity across both ranks: maximum parameter relative error 2.06e-6, momentum relative error 2.87e-6. Full 12-layer runs finish with maximum factor orthogonality error below 9.86e-7 and momentum tangency error below 1.21e-6. These diagnostics are outside the timing and memory window. The combined CPU suite has ten passing tests, including finite-difference transport, MHA/GQA gauge invariance, optimizer/state parity and all scaling geometries.

All 17 successful pilot/correctness JSON artifacts were exported from NFS, checksum-verified and stored under `strict-pilot/`, `cholesky-pilot/`, `checks/`; original-byte hashes are in `sha256.json`. Compiler caches and persisted logs remain on NFS. `jobs.json` records GPU job IDs, source revisions and application outcomes.

## Scaling submissions

Selected policy: reordered + blockwise max-autotune; grouped Tensorion/Riemannian Muon and Cholesky QR/transport, batch size 4 for optimizer grouping; TF32 off. Dense retains both production and grouped Muon controls. No automatic change of precision, batch or checkpointing after OOM.

| Group | Dense-equivalent sizes | Timing attempts | Cloud job |
| --- | --- | ---: | --- |
| small | 257M (12x1024), 411M (24x1024), 823M (12x2048) | 90 | `lm-mpi-job-e2899b0e-b6c5-4aa9-af2b-7bc171e1f2e3` |
| large | 1.439B (24x2048), 2.827B (32x2560) | 60 | `lm-mpi-job-eac45413-0fff-42a6-b3a4-8fbc676ab373` |

Entrypoint arguments: `riemann-scaling small grouped cholesky` / `riemann-scaling large grouped cholesky`. Source: `20119e0d7e8741ecee012249f18e8026523fe06f`. Persistent roots: `/workspace-SR006.nfs3/layerwise-tucker-riemann-20260917/scale-small` and `scale-large`. Each starts with both-rank Cholesky parity gates at its largest width, then enumerates the matrix. Microbatch 1/4/16/32 means accumulation 32/8/2/1, sequence 1024 and 32768 tokens/step; 5 warmup + 20 timed steps, seed 42. Native kernels at all microbatches, matched Liger at microbatch 32. Per-run final geometry checks and explicit OOM/error outcomes are persisted. A launch/status is not successful completion; verify application exits and every expected result before reporting the full matrix.

Startup verified at 2026-09-17 19:37 UTC through completed CPU inspection jobs (IDs in `jobs.json`): both fractions passed the width-2048 and width-2560 gates. Both timing manifests are running. Small has persisted the first two base/microbatch-1 dense controls (reference 446.30 ms, grouped 417.09 ms); large has reached its first dense optimizer step, with no completed timing JSON at this snapshot. The expanded matrix is not complete.

## Storage failure and recovery, 2026-09-18

Both original GPU jobs exited the application with status 1 around 2026-09-17 20:55 UTC, despite Cloud platform status `Completed`. They failed to create result JSON on NFS3. A fresh CPU disk audit (`lm-mpi-job-e94951a6-cfeb-45e0-ace5-a0b3399a1d25`) established the cause: all 500000 NFS3 inodes used, despite 5.8 GiB free. The two scaling directories held about 10.5 GiB of compiler caches versus approximately 6 MiB of logs/results. NFS2 had 6.3 GiB free and 456538 free inodes; node-local `/tmp` had about 305 GiB free.

Recovery job `lm-mpi-job-fe2301b8-46b9-4550-a233-ba692c009db1` copied and SHA256-verified 113 artifact files to `/workspace-SR006.nfs2/layerwise-tucker-riemann-recovery-20260918/original-{small,large}`, then removed only the four old scaling `inductor-cache`/`triton-cache` directories. Logs, correctness checks and result JSON remain intact. Recovery exit 0 was verified. The checksum manifest is retained under `recovery/original-sha256.json` locally. The backup contains 69 small-group successes, 27 large-group successes and 9 OOM outcomes; 45 cases remain.

New GPU jobs are recorded in `jobs.json`: source `0f8421bc04f5f392a31d1d377bef2cba76a60b3c`, NFS2 result roots `scale-small` and `scale-large` under that recovery root. They reuse only matching complete/OOM configurations, retain original source metadata, and rerun missing/invalid results. The numerical/timing protocol is unchanged. Compiler caches and installed temporary dependencies now live in `/tmp` and are removed at application exit; JSON is saved via temporary file and atomic replacement. Recompilation/warmup remains outside timing.

With user approval, the shared platform `logs_dir` was changed from `/home/jovyan/shares/SR006.nfs3/mlsub-logs` to `/home/jovyan/mlsub-logs`, physically NFS1. This does not move or remove old platform logs. NFS1 had 15 GiB and 271723 free inodes at the audit. No unrelated artifacts were deleted.

## Final recovery verification

Both resumed GPU applications exited 0: small finished at 2026-09-18 08:41:59 UTC (platform completion), large at 08:32:46 UTC. The small matrix has 90 successful measurements; large has 33 successful measurements and 27 OOM outcomes. All 105 reused case files are byte-identical to the verified backups. The 45 remaining cases yielded 27 successful measurements and 18 additional OOM outcomes. The model/optimizer source is unchanged between original commit `20119e0` and recovery commit `0f8421b`; the benchmark change is only atomic result-file writing, outside the timed region.

All 156 final JSON artifacts (150 cases, two manifests, four width/rank numerical gates) were downloaded and checksum-verified into `scaling-small/`, `scaling-large/` and their `-checks/` directories. Hashes: `scaling-sha256.json`. Every successful timing file has 5 warmup and 20 finite measured steps, H100 hardware, matching geometry/precision/accumulation, and parameter counts matching its recorded shapes. All four numerical gates passed; maximum full-model factor orthogonality error is 1.55e-6 and momentum tangency error 1.95e-6. These checks establish implementation sanity, not long-training quality equivalence.

Post-cleanup audit `lm-mpi-job-60bab1e9-c6cf-4df5-b415-7e4386e0320b` at 08:27 UTC confirmed 289479 free NFS3 inodes and about 16 GiB free. The old scaling directories now occupy about 5.6 MiB combined; their logs and results remain. New task-local caches/dependencies are removed before application exit. NFS2 retained approximately 6.3 GiB and 456000 free inodes during recovery. A temporary local DNS outage delayed the final download but did not interrupt either GPU job; all final artifacts were subsequently retrieved.
