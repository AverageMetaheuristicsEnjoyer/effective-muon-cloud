# Dense Muon versus layerwise Riemannian Tucker

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
