# Final scaling results

All 150 configurations processed: 123 complete timing runs and 27 CUDA OOM outcomes, no other failures. Both recovery applications exited 0. Reused 105 byte-identical prior results and ran the remaining 45 configurations. All 156 final JSON files (150 cases, two manifests, four numerical gates) were exported with SHA256 verification. Hashes: `scaling-sha256.json`; per-case metrics: `scaling-results.csv`.

Full training step including clipping, optimizer, QR retraction, differential momentum transport, gradient zeroing and synchronization. H100 80GB, sequence 1024, 32768 tokens/step, accumulation 32/8/2/1 for microbatch 1/4/16/32. BF16 autocast, FP32 parameters/state, TF32 off, no activation checkpointing. Five warmup and 20 timed steps, seed 42; compilation/warmup excluded. Reordered contractions and blockwise max-autotune for both dense and Tucker. Liger is enabled equally in its separate rows.

This is a one-seed performance screen, not equal-quality training evidence or a production-TF32 comparison. Dense sizes label architecture dimensions; Tucker parameter counts differ (below). Reused and resumed cases span different H100 nodes; their source commits are retained in every successful JSON. The recovery changed storage/output handling, not the model/optimizer/timed code.

## Speed relative to matched optimized dense

Dense baseline is the faster successful reference/grouped Muon control at the same geometry, microbatch and Liger policy. Ratios are dense time / Tucker time: greater than 1 means faster Tucker. OOM is a capacity outcome, not a speedup.

| Dense geometry | Microbatch | Dense ms/step | A quarter | B quarter | A half | B half |
|---|---:|---:|---:|---:|---:|---:|
| 257M, 12×1024 | 1 | 417.1 | 1.06x | 1.24x | 0.63x | 0.78x |
| 257M, 12×1024 | 4 | 286.6 | 1.21x | 1.47x | 0.61x | 0.81x |
| 257M, 12×1024 | 16 | 257.3 | 1.29x | 1.61x | 0.60x | 0.82x |
| 257M, 12×1024 | 32 | 250.5 | 1.32x | 1.65x | 0.60x | 0.82x |
| 257M, 12×1024 | 32 + Liger | 257.2 | 1.26x | 1.57x | 0.60x | 0.81x |
| 411M, 24×1024 | 1 | 757.9 | 1.07x | 1.25x | 0.61x | 0.76x |
| 411M, 24×1024 | 4 | 522.5 | 1.21x | 1.51x | 0.58x | 0.79x |
| 411M, 24×1024 | 16 | 467.3 | 1.29x | 1.66x | 0.57x | 0.80x |
| 411M, 24×1024 | 32 | 455.0 | 1.32x | 1.72x | 0.57x | 0.79x |
| 411M, 24×1024 | 32 + Liger | 476.0 | 1.26x | 1.59x | 0.57x | 0.79x |
| 823M, 12×2048 | 1 | 1429.4 | 1.45x | 1.87x | 0.54x | 0.72x |
| 823M, 12×2048 | 4 | 1212.4 | 1.63x | 2.23x | 0.53x | 0.76x |
| 823M, 12×2048 | 16 | 1201.1 | 1.74x | 2.44x | 0.55x | 0.80x |
| 823M, 12×2048 | 32 | 1191.7 | 1.76x | 2.48x | 0.56x | 0.80x |
| 823M, 12×2048 | 32 + Liger | 1203.8 | 1.72x | 2.40x | 0.56x | 0.80x |
| 1.44B, 24×2048 | 1 | 2734.6 | 1.47x | 1.93x | 0.52x | 0.71x |
| 1.44B, 24×2048 | 4 | 2341.0 | 1.65x | 2.33x | 0.53x | 0.76x |
| 1.44B, 24×2048 | 16 | 2254.4 | 1.72x | 2.48x | 0.53x | 0.77x |
| 1.44B, 24×2048 | 32 | OOM | OOM | OOM | OOM | OOM |
| 1.44B, 24×2048 | 32 + Liger | 2260.8 | OOM | 2.40x | OOM | OOM |
| 2.83B, 32×2560 | 1 | 6551.0 | 1.67x | 2.25x | 0.51x | 0.71x |
| 2.83B, 32×2560 | 4 | 5848.5 | 1.89x | 2.74x | 0.52x | 0.76x |
| 2.83B, 32×2560 | 16 | OOM | OOM | OOM | OOM | OOM |
| 2.83B, 32×2560 | 32 | OOM | OOM | OOM | OOM | OOM |
| 2.83B, 32×2560 | 32 + Liger | OOM | OOM | OOM | OOM | OOM |

## Full-step median times, ms

| Dense geometry | Microbatch | Dense | A quarter | B quarter | A half | B half |
|---|---:|---:|---:|---:|---:|---:|
| 257M, 12×1024 | 1 | 417.1 | 392.6 | 337.6 | 658.3 | 532.6 |
| 257M, 12×1024 | 4 | 286.6 | 237.3 | 195.5 | 472.6 | 355.8 |
| 257M, 12×1024 | 16 | 257.3 | 200.2 | 159.8 | 427.7 | 312.4 |
| 257M, 12×1024 | 32 | 250.5 | 190.1 | 151.7 | 417.7 | 304.5 |
| 257M, 12×1024 | 32 + Liger | 257.2 | 203.5 | 164.3 | 428.2 | 316.6 |
| 411M, 24×1024 | 1 | 757.9 | 708.5 | 607.7 | 1249.0 | 998.5 |
| 411M, 24×1024 | 4 | 522.5 | 431.4 | 345.0 | 897.2 | 664.8 |
| 411M, 24×1024 | 16 | 467.3 | 363.5 | 281.1 | 817.6 | 586.9 |
| 411M, 24×1024 | 32 | 455.0 | 345.2 | 265.2 | 799.0 | 572.6 |
| 411M, 24×1024 | 32 + Liger | 476.0 | 377.6 | 299.2 | 828.0 | 600.7 |
| 823M, 12×2048 | 1 | 1429.4 | 986.1 | 764.9 | 2671.4 | 1986.7 |
| 823M, 12×2048 | 4 | 1212.4 | 742.6 | 544.0 | 2268.9 | 1601.5 |
| 823M, 12×2048 | 16 | 1201.1 | 691.8 | 491.5 | 2164.2 | 1504.7 |
| 823M, 12×2048 | 32 | 1191.7 | 679.0 | 480.7 | 2146.2 | 1485.7 |
| 823M, 12×2048 | 32 + Liger | 1203.8 | 700.0 | 501.3 | 2166.0 | 1506.8 |
| 1.44B, 24×2048 | 1 | 2734.6 | 1864.5 | 1420.1 | 5229.4 | 3866.1 |
| 1.44B, 24×2048 | 4 | 2341.0 | 1415.1 | 1003.7 | 4420.0 | 3094.8 |
| 1.44B, 24×2048 | 16 | 2254.4 | 1308.3 | 908.9 | 4231.9 | 2919.9 |
| 1.44B, 24×2048 | 32 | OOM | OOM | OOM | OOM | OOM |
| 1.44B, 24×2048 | 32 + Liger | 2260.8 | OOM | 942.0 | OOM | OOM |
| 2.83B, 32×2560 | 1 | 6551.0 | 3929.8 | 2905.4 | 12904.1 | 9256.1 |
| 2.83B, 32×2560 | 4 | 5848.5 | 3094.8 | 2137.5 | 11219.9 | 7649.3 |
| 2.83B, 32×2560 | 16 | OOM | OOM | OOM | OOM | OOM |
| 2.83B, 32×2560 | 32 | OOM | OOM | OOM | OOM | OOM |
| 2.83B, 32×2560 | 32 + Liger | OOM | OOM | OOM | OOM | OOM |

## Parameters, millions

| Geometry | Dense | A quarter | B quarter | A half | B half |
|---|---:|---:|---:|---:|---:|
| 257M, 12×1024 | 257.19 | 169.75 | 142.81 | 255.71 | 201.84 |
| 411M, 24×1024 | 411.35 | 236.47 | 182.60 | 408.41 | 300.66 |
| 823M, 12×2048 | 822.66 | 472.75 | 365.01 | 816.47 | 600.98 |
| 1.44B, 24×2048 | 1439.27 | 739.45 | 523.96 | 1426.89 | 995.92 |
| 2.83B, 32×2560 | 2826.73 | 1368.69 | 919.77 | 2800.78 | 1902.94 |

B-quarter wins all comparable completed points; half-rank variants are slower than dense at every comparable completed point. At 1.439B/microbatch 32, native kernels OOM for all arms, while matched Liger permits dense and B-quarter (2.40x speedup). At 2.827B, microbatches 16 and 32 OOM for every arm in this protocol, including the tested microbatch-32 Liger cases.

Storage recovery removed about 10.5 GiB of owned scaling compiler caches after verified backups, leaving source logs/results intact. NFS3 free inodes went from 0 to about 289000, with 16 GiB free in the post-cleanup audit. Recovery results are persisted on NFS2; new compiler caches/dependencies were node-local and removed at application exit.
