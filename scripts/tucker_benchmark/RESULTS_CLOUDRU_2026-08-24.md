# Static Tucker: forward/backward и Muon на Cloud.ru

Ветка `tucker-membench`, commit `69e569c`. Все GPU-точки измерены на NVIDIA
H100 80GB HBM3 в BF16. Модель — Llama 257M, sequence length 1024. На каждой
capacity-точке обрабатывается 16,384 токена; выполнено 3 warmup и 12 измеряемых
шагов.

В отличие от первого отчёта, основная метрика здесь — сумма CUDA Event-времён
`forward + backward`, без gradient clipping, optimizer step, QR-retraction и
vector transport. Сравниваются Dense AdamW, Dense Muon и Static Tucker +
Tensorion. У Tucker core и четыре factor direction вычисляются в пяти CUDA
streams; dense Muon использует 5 Newton–Schulz steps, Tensorion — 6.

## Forward + backward, мс

| Microbatch | Accumulation | Dense AdamW | Dense Muon | Static Tucker | Tucker / Muon |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 360.727 | 368.460 | 1318.437 | 3.58x |
| 2 | 8 | 178.151 | 177.777 | 651.523 | 3.67x |
| 4 | 4 | 118.167 | 118.627 | 329.870 | 2.78x |
| 8 | 2 | 113.306 | 113.372 | 209.822 | 1.85x |
| 16 | 1 | 106.843 | 107.786 | 198.285 | 1.84x |

Dense AdamW и Dense Muon дают практически одинаковое forward/backward-время.
Следовательно, optimizer backend не объясняет более медленный Tucker forward:
разница находится в factorized contraction и её backward.

## Optimizer section, мс

| Microbatch | Dense AdamW | Dense Muon | Static Tucker, 5 streams | Tucker sequential, старый прогон |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 1.834 | 42.787 | 586.684 | 444.376 |
| 2 | 1.845 | 40.562 | 582.271 | 424.360 |
| 4 | 1.845 | 33.526 | 585.267 | 398.123 |
| 8 | 1.846 | 34.269 | 574.778 | 446.803 |
| 16 | 1.845 | 34.747 | 574.718 | 439.373 |

Пять CUDA streams не ускорили optimizer: measured section стала медленнее
старого последовательного варианта на 29–47% в capacity sweep. Значит,
предположение «стоимость возникает только из-за пяти последовательных Muon» не
подтвердилось. В Tucker optimizer также входят coupled LR scaling со spectral
estimates, QR-retraction и vector transport; четыре маленьких factor NS не
доминируют над core и остальными операциями.

## Полный шаг и peak CUDA memory

| Microbatch | AdamW step, ms | Muon step, ms | Tucker step, ms | AdamW peak, GB | Muon peak, GB | Tucker peak, GB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 364.249 | 413.018 | 1909.140 | 3.267 | 2.965 | 4.234 |
| 2 | 181.508 | 219.773 | 1237.386 | 4.408 | 4.100 | 6.490 |
| 4 | 120.864 | 152.999 | 918.959 | 6.610 | 6.284 | 10.727 |
| 8 | 116.008 | 148.450 | 786.563 | 11.021 | 10.694 | 19.417 |
| 16 | 109.514 | 143.379 | 775.311 | 19.279 | 18.976 | 36.262 |

## Production point: microbatch 32 × accumulation 4

На optimizer step приходится 131,072 токена.

| Variant | Forward + backward, ms | Optimizer, ms | Full step, ms | Peak, GB |
| --- | ---: | ---: | ---: | ---: |
| Dense AdamW | 821.571 | 1.750 | 824.186 | 37.441 |
| Dense Muon | 827.650 | 35.009 | 863.574 | 37.136 |
| Static Tucker, 5 streams | 1499.104 | 571.316 | 2073.215 | 71.539 |

В production-точке Tucker forward + backward в 1.81 раза медленнее Dense Muon
и в 1.82 раза медленнее Dense AdamW. Параллельный optimizer в 1.21 раза
медленнее старого последовательного результата 473.725 ms.

Все 84 Tucker-слоя работали в режиме `contract`. CPU self-test заменял
`materialize_weight` на исключение и успешно выполнил forward, backward,
Tensorion step и retraction.

Cloud jobs:

- CPU self-test: `lm-mpi-job-7d396e03-59fd-40f2-a4c4-302b8298e961`
- H100 smoke: `lm-mpi-job-412fa561-416e-4e43-8d5a-608e97f2aef3`
- Capacity sweep: `lm-mpi-job-0583f75d-0fa8-4f7b-8e5e-18d8fae2f90a`
- Production point: `lm-mpi-job-f405573b-c021-46cc-bad7-5103d709b023`

Raw JSON сохранён в
`/workspace-SR006.nfs3/tucker-membench-parallel/results` и
`/workspace-SR006.nfs3/tucker-membench-parallel-production/results`.
