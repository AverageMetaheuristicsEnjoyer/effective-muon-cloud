# Fused direct Tucker с Dense lm_head

Это изолированный эксперимент ускорения 257M Tucker Llama без материализации
полных dense-весов для внутренних Tucker-слоёв.

## Архитектурный контракт

- `lm_head` всегда обычный Dense `nn.Linear(1024, 50304, bias=False)`;
- Tucker заменяет ровно 84 внутренние матрицы: по Q/K/V/O/Gate/Up/Down в
  каждом из 12 блоков;
- Tucker-модель содержит ровно **257,676,352** параметра;
- Dense control содержит **257,188,864** параметра;
- параметры не удаляются, не замораживаются и не объединяются.

Скрипты benchmark содержат runtime-проверки этих условий. Число 257,193,298 и
старые измерения с Tucker `lm_head` являются другой, ошибочной для этой задачи
архитектурой и здесь не используются.

## Реализованные оптимизации

`tucker_fused_ops.py` подменяет динамические вызовы в
`models.tucker_chunked`:

1. Triton input/output mode-pair kernels работают на полном внутреннем чанке
   16,384 токена вместо искусственного ограничения в 1,024;
2. BF16 work-копии FP32 Tucker-параметров сохраняются между forward и backward;
3. cache автоматически инвалидируется после in-place optimizer update;
4. paired analytical VJP kernels объединяют обе mode-gradient операции и
   сокращают recompute/copy tax в backward.

Оптимизация относится только к 84 внутренним Tucker-модулям. Dense `lm_head`
и fused Liger linear cross entropy не меняются.

## Результат на A100

Одна A100 PCIe 40 GB, BF16, batch 16, sequence 1024, forward + backward без
optimizer:

| Реализация | Median | Peak allocated |
|---|---:|---:|
| Baseline direct Tucker, Dense head | 516.9 ms | 8840.3 MiB |
| Fused analytical backward, Dense head | **418.2 ms** | **8752.3 MiB** |

Это ускорение 19.1% и экономия 88 MiB при неизменных 257,676,352 параметрах.
Для ориентира, чистый Dense control того же размера блоков: 236.5 ms и
8407.8 MiB. Таким образом, текущий Tucker всё ещё в 1.77 раза медленнее Dense и
использует на 344.5 MiB больше peak allocated memory.

Полный fused Tucker + Muon step с Dense head использует 10134.9 MiB peak и
1376.1 MiB optimizer state. Dense + Muon использует 9792.6 MiB и 1374.2 MiB:
разница peak составляет 342.3 MiB. Чистое время Tucker+Muon пока не зафиксировано:
на общей карте повторные запуски дали нестабильные 965.7--2036.0 ms median,
поэтому эти wall-time значения не считаются результатом производительности.

## Команды

Parity внутренних Tucker kernels:

```bash
CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/test_fused_ops.py
```

Сравнение baseline и fused backward на одной и той же модели:

```bash
CUDA_VISIBLE_DEVICES=<A100_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/benchmark_ablation.py \
  --batch-size 16 --sequence-length 1024 --warmup 1 --rounds 5
```

Полное обучение:

```bash
bash run_fused_dense_head_tucker.sh
```

`TUCKER_FUSED_BACKWARD=0` отключает custom backward для контрольного запуска.
`TUCKER_ONLINE_CE=1` намеренно запрещён target entrypoint, так как online Tucker
CE несовместим с контрактом Dense `lm_head`.

## Исследовательские файлы вне target path

`tucker_online_ce.py` и `test_online_head_parity.py` сохранены как отдельный
прототип Tucker-head. Они не вызываются launcher-ом и исключены из всех
актуальных сравнений.

CUDA Graph также не является рекомендуемым режимом: после устранения первых
host sync capture всё ещё блокируется в PyTorch/CUB backward, а replay
потребовал бы явного обновления BF16 cache после optimizer step.

На общем сервере перед измерением необходимо убедиться, что выбранная A100
свободна. Memory peak воспроизводится стабильно, но сторонняя GPU-нагрузка
сильно искажает wall time.
