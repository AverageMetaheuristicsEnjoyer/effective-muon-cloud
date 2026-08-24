# Cloud.ru H100 benchmark results

Branch `tucker-membench`, commit `3cdf678`. The GPU was an NVIDIA H100 80GB
HBM3. Times are median host wall times for a complete optimizer step. Memory is
peak CUDA allocated memory in decimal GB.

The capacity sweep used sequence length 1024, 16,384 tokens per optimizer step,
3 warmup steps and 12 measured steps.

| Microbatch | Accumulation | Dense ms | Tucker ms | Dense tok/s | Tucker tok/s | Dense peak GB | Tucker peak GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 264.095 | 1675.080 | 62,038 | 9,781 | 3.267 | 4.057 |
| 2 | 8 | 153.137 | 968.180 | 106,990 | 16,923 | 4.408 | 6.325 |
| 4 | 4 | 120.581 | 645.391 | 135,876 | 25,396 | 6.610 | 10.560 |
| 8 | 2 | 115.263 | 659.110 | 142,144 | 24,858 | 11.021 | 19.258 |
| 16 | 1 | 109.955 | 638.757 | 149,006 | 25,651 | 19.279 | 36.086 |

Optimizer state was 1.029 GB for dense AdamW and 0.723 GB for static Tucker.
Model storage was 0.514 GB and 0.515 GB respectively. At microbatch 16, static
Tucker was 5.81x slower and used 1.87x the peak allocated memory. Its median
optimizer section, including retraction and vector transport, was 439.373 ms,
compared with 1.846 ms for dense AdamW.

The launcher-equivalent production point used microbatch 32, accumulation 4,
and 131,072 tokens per optimizer step:

| Variant | Median ms | Tokens/s | Peak GB | Optimizer state GB | Optimizer ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dense AdamW | 823.389 | 159,186 | 37.441 | 1.029 | 1.753 |
| Static Tucker | 1973.803 | 66,406 | 71.364 | 0.723 | 473.725 |

The static model had exactly 257,676,352 parameters. All 84 Tucker layers
reported `contract`; the self-test also replaced `materialize_weight` with an
exception while running forward, backward, optimizer step and retraction.

Cloud jobs:

- Full capacity sweep: `lm-mpi-job-2fe89f50-a619-450c-a2f2-b09f7ca2c6a2`
- Production point: `lm-mpi-job-a8578193-cca9-46ea-a4c3-464fdb1d5a24`
- No-materialization self-test: `lm-mpi-job-c22dd348-6446-40b5-b6b8-30cca7b13feb`

Raw JSON remains on Cloud.ru under
`/workspace-SR006.nfs3/tucker-membench/results` and
`/workspace-SR006.nfs3/tucker-membench-production/results`.
