# Custom Tucker backward: итог autoresearch

Это изолированный production-path для 84 внутренних `TuckerLinear` модели.
`lm_head` во всех тестах и launcher остаётся обычным
`nn.Linear(1024, 50304, bias=False)` и обучается через Dense Liger fused linear
cross-entropy. Внутренний полный Dense-вес Tucker нигде не создаётся.

## Одна команда запуска

```bash
bash experiments/fused_persistent_tucker/custom_backward/run_final.sh
```

По умолчанию включены:

- `hybrid_gate_up` BF16 cache — лучший баланс времени и памяти;
- custom analytical backward и layout-aware Triton VJP;
- stream-parallel Muon для всех 84 Tucker-ядер и grouped Muon для всех 336
  `U1…U4` факторов; Dense `lm_head` AdamW перекрывается с ними;
- grouped QR retraction;
- Dense `lm_head`; `TUCKER_ONLINE_CE=0` принудительно.

Для минимальной памяти можно запустить
`TUCKER_CUSTOM_CACHE_POLICY=recast bash .../run_final.sh`. Для ablation grouped
частей существуют `TUCKER_PARALLEL_MUON=0` и
`TUCKER_GROUPED_RETRACTION=0`. Выбранные A100 defaults можно переопределить
через `TUCKER_MUON_CORE_MICROBATCH` и `TUCKER_MUON_STREAMS`.

## Зафиксированная модель

| Проверка | Результат |
|---|---:|
| Trainable parameters | 257,676,352 |
| Tucker modules | 84 = 12 × 7 |
| Tucker/other parameters | 154,628,160 / 103,048,192 |
| `lm_head` | Dense `nn.Linear`, 51,511,296 parameters |
| Internal Dense materialization | нет |
| Full logits `[16384,50304]` | нет, используется Dense Liger CE |

Production shapes: 48× `1024→1024` ranks `(32,32,32,32)`, 24×
`1024→2816` ranks `(32,32,44,64)`, 12× `2816→1024` ranks
`(44,64,32,32)`.

## Что оптимизировано

1. Backward больше не пересчитывает отброшенный результат через `U4`.
2. Одночанковый hot path сразу возвращает собственный `dX` и parameter grads:
   без второго full-activation copy и FP32 zero/add accumulators.
3. Два Triton VJP-kernel сразу записывают `grad_third`, `grad_first`, input и
   `core_out` в layout, который нужен следующим GEMM. Удалено 168 отдельных
   layout-copy на backward.
4. A100 autotune выбрал `num_warps=4`, `num_stages=4` для обеих production
   families. Полные samples лежат в `results/autotune_a100.json`.
5. `hybrid_gate_up` хранит BF16 work-copy только 24 самых крупных Gate/Up
   Tucker-модулей: 132.37 MiB cache вместо полного persistent cache.
6. Muon объединяет Newton–Schulz для 336 одинаковых малых факторов, а 84
   независимых Tucker-ядра распределяет по двум CUDA streams. AdamW для
   embedding, norm и **Dense `lm_head`** одновременно выполняется на caller
   stream. Математическая формула Muon/AdamW не меняется.
7. QR gauge fixing выполняет batched QR и batched core mode-products по трём
   реальным rank groups.

## A100 результаты: forward + backward

Условия: A100 PCIe 40 GB, BF16, B=16, S=1024, Dense head, одинаковые inputs,
10 warmup и 30 measured для финального interleaved прогона. Время — median.

| Реализация | F+B, ms | Peak allocated, MiB |
|---|---:|---:|
| Dense reference (исторический корректный control) | 236.5 | 8407.8 |
| Tucker reference direct | 548.857 | 8550.0 |
| Предыдущий fused Tucker | 416.440 | 8752.8 |
| Custom persistent | **369.107** | 8501.8 |
| Custom `hybrid_gate_up` — default | 370.320 | **8343.4** |
| Custom recast — minimum memory | 370.897 | **8211.0** |

Default ускоряет direct Tucker на 32.5%, предыдущий fused path — на 11.1%, и
использует на 64.4 MiB меньше peak памяти, чем Dense reference. Recast ещё на
132.4 MiB экономнее при разнице около 0.6 ms. Все Tucker-варианты имеют ровно
257,676,352 параметра; Dense control — 257,188,864.

Финальный single-step profile (`results/final_recast_profile.json`):

| Region | CUDA time |
|---|---:|
| Tucker custom backward, 84 вызова | 156.31 ms |
| Все `aten::mm`, 800 вызовов | 146.56 ms |
| Layout/cast copies, 1652 вызова | 47.85 ms |
| Два transposed Triton VJP | 35.66 ms |
| Dense Liger head + CE | 61.73 ms |

Цель внутреннего Tucker backward 120–170 ms достигнута; цель полного F+B
300–400 ms также достигнута. После layout folding один profile step снизился с
402.67 до 375.62 ms. Chrome traces до/после сохранены в `results/`.

## Полный Muon step и память

Чистый run обычного Muon без retraction (3 warmup + 10 measured): forward
134.142 ms, backward 223.082 ms, clip 8.190 ms, optimizer 452.696 ms, полный
шаг 818.107 ms. Training peak после создания optimizer state — 9725.47 MiB.

Размеры постоянных данных:

| Буфер | MiB | dtype |
|---|---:|---|
| Model parameters | 982.957 | FP32 |
| Gradients | 982.957 | FP32 |
| Muon + AdamW state | 1376.055 | FP32 |
| Hybrid BF16 work cache | 132.370 | BF16 |

Предыдущий grouped-small результат не является окончательной оценкой Muon: он
группировал 336 факторов, но оставлял все 84 core matrices последовательными.
Теперь параллелятся обе независимые семьи, причём AdamW Dense head тоже
перекрывается с Muon.

Изолированный same-process autotune (`3 warmup + 10 measured`, одна модель и
одни gradients для всех кандидатов):

| Muon schedule | Optimizer, median ms | p10…p90, ms | Phase peak, MiB |
|---|---:|---:|---:|
| Последовательный reference | 554.884 | 531.446…593.421 | 4271.65 |
| Только 336 grouped factors | 506.221 | 372.301…616.123 | 4271.52 |
| Все factors + cores, 1 stream | 301.939 | 296.196…312.122 | **4252.55** |
| Все factors + cores, 2 streams — selected | **269.267** | **263.915…274.840** | 4270.02 |
| Все factors + cores, 4 streams | 274.841 | 262.583…284.000 | 4303.52 |

Selected schedule на 51.5% быстрее последовательного reference в этом
same-process sweep и не добавляет optimizer memory. `core_microbatch=1`
оказался быстрее 2/4/8: крупные cores выгоднее запускать независимо на двух
streams, а не собирать в batched GEMM. Все 14 кандидатов сохранены в
`results/parallel_muon_autotune_a100.json`.

Финальный полный `10 warmup + 30 measured` с selected Muon и grouped QR:
optimizer **262.651 ms**, grouped cores/factors `84/336`, optimizer-phase peak
4369.15 MiB, global training peak 9770.72 MiB. На общей карте F+B испытывал
сильную переменную внешнюю нагрузку, поэтому полный wall 839.34 ms не считается
чистым throughput-результатом; optimizer-only autotune выше имеет узкий
p10…p90 и используется для выбора production-конфигурации.

Трёхшаговый parity-тест сравнивает параметры и momentum с исходным Muon, а
также параметры, first/second moments исходного AdamW. Он включает Dense-like
`lm_head.weight` и resume из vanilla-compatible optimizer checkpoint; все
проверки пройдены.

Grouped QR retraction прошёл проверку output invariance и ортогональности.
Probe: 89.84 ms против 160.60 ms последовательного пути. Его временный peak
4520.52 MiB ниже общего training peak 9722.97 MiB, поэтому глобального memory
regression нет.

## Почему grouped QKV/Gate-Up не включён

Microbenchmark показал небольшой выигрыш forward/dX, но batched `dW` оказался
медленнее. Суммарно QKV теряет около 0.010 ms на слой, Gate/Up — около 0.030
ms на слой; BF16 grouped `dW` также меняет reduction order (relative L2
2.6e-3…3.9e-3). Поэтому этот кандидат сознательно не интегрирован.

## Проверки и воспроизведение

```bash
CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_correctness.py

CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_parallel_muon.py

CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_grouped_retraction.py

CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/benchmark_full.py \
  --warmup 10 --rounds 30

CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/benchmark_muon.py \
  --cache-policy hybrid_gate_up --parallel-grouped-muon \
  --core-microbatch 1 --muon-streams 2 \
  --retract --grouped-retraction --warmup 10 --iterations 30

CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/autotune_parallel_muon.py \
  --warmup 3 --iterations 10
```

Correctness suite покрывает forward, loss, `dX`, все пять Tucker gradients,
все production shapes, non-contiguous input, multi-chunk, accumulation,
cold/warm cache, invalidation после optimizer update, NaN/Inf и несколько
training steps. Все параметры, включая Dense `lm_head`, получают finite grads.

Environment: PyTorch 2.7.1+cu118, Triton 3.3.1, Liger Kernel 0.8.1, BF16,
NVIDIA A100. Исходный path сохранён как reference/fallback.
