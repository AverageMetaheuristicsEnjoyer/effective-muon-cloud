# Handoff: memory/time-optimized Tucker training with Dense `lm_head`

Дата состояния: 26 августа 2026. Этот документ предназначен для передачи
проекта другому Codex/инженеру без истории текущего чата.

## 1. Коротко: что передаётся

Production-конфигурация обучает Llama с:

- ровно 84 внутренними Tucker-матрицами: Q/K/V/O/Gate/Up/Down в 12 слоях;
- обычным Dense `nn.Linear(1024, 50304, bias=False)` в `lm_head`;
- 257,676,352 trainable parameters;
- отсутствием материализации полных Dense-весов внутренних Tucker-слоёв;
- FP32 master parameters, FP32 gradients и FP32 optimizer states;
- BF16 forward/backward, как в исходном training regime;
- analytical Tucker backward;
- stream-parallel Muon для Tucker cores/factors;
- grouped QR retraction после optimizer step.

Ничего не квантизовано в FP8/INT8, параметры не удалены, не заморожены и не
объединены. Математика forward/backward, Muon и AdamW сохранена. Возможны
обычные небольшие floating-point отличия из-за другого порядка batched GEMM.

## 2. Готовый prompt для следующего Codex

Передай ему всю папку или архив и этот текст:

> Открой `TUCKER_PARALLEL_HANDOFF.md` и выполни его как основной контракт.
> Не меняй Dense `lm_head` на Tucker. Не материализуй полные Dense-веса
> внутренних Tucker-слоёв. Сначала воспроизведи correctness tests и текущий
> benchmark, затем адаптируй shape/rank plan и autotune для целевой модели/GPU.
> Все сравнения должны использовать одинаковые batch, sequence, dtype, loss,
> optimizer и Dense head. Не выдавай результаты с занятой GPU за clean timing.

Рекомендуемый порядок чтения:

1. этот документ;
2. `experiments/fused_persistent_tucker/custom_backward/README.md`;
3. `experiments/fused_persistent_tucker/custom_backward/ops.py`;
4. `experiments/fused_persistent_tucker/custom_backward/parallel_muon.py`;
5. `experiments/fused_persistent_tucker/custom_backward/train_entry.py`;
6. целевой launcher в `scripts/single_gpu/tucker_transformer/`.

## 3. Где лежит production path

Главная команда:

```bash
bash run_fused_dense_head_tucker.sh
```

Она вызывает:

```text
run_fused_dense_head_tucker.sh
  -> experiments/fused_persistent_tucker/custom_backward/train_entry.py
  -> scripts/single_gpu/tucker_transformer/
       fineweb_standard_attention_muon_tucker_retract_1x_chinchilla.sh
  -> src/main.py
```

Основные файлы:

| Файл | Назначение |
|---|---|
| `custom_backward/integration.py` | Подмена только `chunked_tucker_linear`; исходный reference остаётся доступен. |
| `custom_backward/ops.py` | Custom autograd Function, cache policies и analytical backward. |
| `custom_backward/kernels.py` | Layout-aware Triton VJP kernels. |
| `custom_backward/parallel_muon.py` | Parallel Muon для cores/factors и overlap Dense-head AdamW. |
| `custom_backward/grouped_retraction.py` | Batched QR и batched core mode-products. |
| `custom_backward/train_entry.py` | Устанавливает production monkey patches перед запуском `src/main.py`. |
| `custom_backward/run_final.sh` | Финальные environment defaults. |
| `src/models/tucker_linear.py` | Tucker-модуль, rank planning, replacement и reference QR. |
| `src/models/tucker_chunked.py` | Correctness reference для chunked forward/backward. |
| `src/models/tucker_triton.py` | Исходные fused mode-pair Triton kernels. |

`tucker_online_ce.py` не относится к production path. Он был прототипом для
Tucker head. В текущем контракте `lm_head` всегда Dense, поэтому
`TUCKER_ONLINE_CE=1` намеренно вызывает ошибку.

## 4. Что именно было изменено

### 4.1 Analytical backward

Исходный direct Tucker вычисляет:

```text
x -> U1 -> U2 -> core -> U3 -> U4
```

Custom backward:

- не пересчитывает логический output через `U4` второй раз;
- пересчитывает только VJP intermediates;
- для одного token chunk возвращает собственные `dX`, `dCore`, `dU1..dU4`
  без второго FP32 accumulation buffer;
- сразу записывает несколько intermediates в GEMM-ready transposed layouts;
- при нескольких chunks использует безопасное FP32 accumulation.

### 4.2 Cache policies

Доступны:

- `persistent`: BF16 work copies всех Tucker-параметров сохраняются между
  forward и backward;
- `recast`: BF16 work copies заново создаются в backward; минимум памяти;
- `hybrid_gate_up`: текущий A100 default, сохраняет только самые дорогие
  Gate/Up family текущей 257M модели.

Cache инвалидируется по `parameter._version` после optimizer update. Master
parameters остаются FP32; cache — лишь BF16 working representation, которую
обычный autocast всё равно использует в GEMM.

Важно для другой модели: текущий `hybrid_gate_up` содержит эвристику
`module.out_features == 2816`. Для другой FFN width она, скорее всего, не
выберет нужные слои. До обобщения используй `recast` или `persistent`, затем
замени эвристику на явный per-module cache plan/бюджет.

### 4.3 Parallel Muon

Текущая реализация:

- находит параметры по именам `.core_matrix` и `.U1`...`.U4`;
- объединяет одинаковые factors по `(optimizer group, kind, shape, device)`;
- запускает независимые core updates на worker CUDA streams;
- запускает исходный AdamW path, включая Dense `lm_head`, на caller stream;
- ждёт workers перед QR retraction и следующим training step;
- не меняет momentum, Nesterov, Newton–Schulz, decay или update scaling;
- сохраняет checkpoint с обычным `use_muon=2`, а после load восстанавливает
  transient parallel routing. Старые vanilla-Muon checkpoints совместимы.

A100 sweep выбрал:

```text
core_microbatch = 1
worker streams  = 2
factor batching = все одинаковые factors одной shape family
```

Батчирование больших cores по 2/4/8 было медленнее. Отдельные cores выгоднее
распределять между streams; маленькие factors выгоднее складывать в batch.

Parallel path применяется к vanilla-Muon параметрам (`use_muon == 2`). Если
включить другую LITE/Riemannian маршрутизацию, сначала проверь, какие параметры
получают `use_muon == 1`; они останутся на исходном последовательном path.

### 4.4 Grouped QR

Модули группируются по `(modes, ranks)`. Для каждой группы выполняются batched
QR и batched core mode-products. Представляемый линейный оператор сохраняется.

Если включён `transport_optimizer_state=True`, grouped implementation
намеренно возвращается к reference path. Текущий launcher vector transport не
включает.

## 5. Текущая архитектура и dtype contract

```text
model                    Llama
layers                   12
hidden                   1024
heads                    8
FFN hidden               2816
vocabulary               50304
batch per microstep      16
sequence                 1024
tokens per microstep     16384
Tucker modules           84
Tucker rank CLI          259 (clamped independently to each mode)
parameters               257,676,352
Dense lm_head parameters 51,511,296
model/master dtype       FP32
gradient dtype           FP32
optimizer state dtype    FP32
compute/autocast dtype    BF16
```

Production shape families:

```text
48 x 1024 -> 1024, ranks (32, 32, 32, 32)
24 x 1024 -> 2816, ranks (32, 32, 44, 64)
12 x 2816 -> 1024, ranks (44, 64, 32, 32)
```

`lm_head` не входит в эти 84 модуля.

## 6. Воспроизведение на существующем A100-сервере

Проект на сервере:

```bash
cd /home/rustikkabirov/projects/tucker_fused_a100_20260825
```

Не сохраняй SSH password/token в документации или коде. Выбери физический GPU
UUID и сначала проверь внешнюю нагрузку:

```bash
nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

Correctness:

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_correctness.py

CUDA_VISIBLE_DEVICES=<GPU_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_parallel_muon.py

CUDA_VISIBLE_DEVICES=<GPU_UUID> .venv-a100/bin/python \
  experiments/fused_persistent_tucker/custom_backward/test_grouped_retraction.py
```

Training:

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> \
PYTHON_BIN=.venv-a100/bin/python \
WANDB_MODE=offline \
bash run_fused_dense_head_tucker.sh
```

Для online logging задай `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT` и
при необходимости `WANDB_BASE_URL`. FineWeb path управляется `DATASETS_DIR`.

Первый step существенно медленнее: компилируются Triton/Inductor kernels и
создаются optimizer states. Не включай его в steady-state benchmark.

## 7. Установка на другой сервер / H100

Передавай всю папку или архив
`tucker_dense_head_parallel_muon_FINAL.zip`. В архиве нет dataset, checkpoints,
W&B credentials и CUDA caches.

Вариант свежей установки:

```bash
unzip tucker_dense_head_parallel_muon_FINAL.zip -d tucker_project
cd tucker_project
conda env create -f environment.yml
conda activate huawei-stage2
bash setup.sh
```

`setup.sh` сейчас выбирает PyTorch 2.9 wheel по установленному CUDA toolkit и
требует CUDA >= 12.6. Валидированный A100 server environment был PyTorch
2.7.1+cu118, Triton 3.3.1, Liger 0.8.1. Оба варианта допустимы, но после любой
смены PyTorch/Triton обязательно заново запусти correctness и autotune.

H100 поддерживается: здесь нет A100-specific assembly. A100-selected configs
корректно исполнятся, но могут быть не оптимальны для SM90.

```bash
CUDA_VISIBLE_DEVICES=<H100_UUID> \
bash run_fused_dense_head_tucker.sh
```

После smoke test заново настрой kernels и Muon streams, как описано ниже.

## 8. Production environment switches

| Переменная | Default | Значение |
|---|---:|---|
| `TUCKER_CUSTOM_CACHE_POLICY` | `hybrid_gate_up` | `persistent`, `recast`, `hybrid_gate_up`. |
| `TUCKER_PARALLEL_MUON` | `1` | Полноценный core/factor parallel Muon. |
| `TUCKER_MUON_CORE_MICROBATCH` | `1` | Сколько equal-shaped cores складывать в один NS batch. |
| `TUCKER_MUON_STREAMS` | `2` | Число worker CUDA streams. |
| `TUCKER_GROUPED_SMALL_MUON` | `0` | Старый factor-only fallback; не включать одновременно с parallel path. |
| `TUCKER_GROUPED_RETRACTION` | `1` | Batched QR retraction. |
| `TUCKER_ONLINE_CE` | `0` | Должно оставаться `0`, так как head Dense. |
| `TUCKER_CONTRACT_CHUNK_SIZE` | `16384` | Token chunk во внутреннем Tucker forward/backward. |
| `ACTIVATION_CHECKPOINTING` | `0` | Можно включить ценой времени, если памяти не хватает. |
| `TORCH_COMPILE` | `0` | Не был выбран как production default. |

Minimum-memory пример:

```bash
TUCKER_CUSTOM_CACHE_POLICY=recast bash run_fused_dense_head_tucker.sh
```

Ablation без parallel Muon:

```bash
TUCKER_PARALLEL_MUON=0 \
TUCKER_GROUPED_SMALL_MUON=0 \
bash run_fused_dense_head_tucker.sh
```

## 9. Зафиксированные A100 результаты

Условия: A100 PCIe 40 GB, BF16, batch 16, sequence 1024, Dense head.

Forward + backward:

| Path | Median | Peak allocated |
|---|---:|---:|
| Dense control | 236.5 ms | 8407.8 MiB |
| Direct Tucker reference | 548.857 ms | 8550.0 MiB |
| Custom persistent | 369.107 ms | 8501.8 MiB |
| Custom `hybrid_gate_up` | 370.320 ms | 8343.4 MiB |
| Custom `recast` | 370.897 ms | 8211.0 MiB |

Muon optimizer-only same-process sweep:

| Schedule | Median | p10...p90 | Phase peak |
|---|---:|---:|---:|
| Sequential | 554.884 ms | 531.446...593.421 | 4271.65 MiB |
| Factor-only grouped | 506.221 ms | 372.301...616.123 | 4271.52 MiB |
| Cores + factors, 1 stream | 301.939 ms | 296.196...312.122 | 4252.55 MiB |
| Cores + factors, 2 streams | **269.267 ms** | **263.915...274.840** | 4270.02 MiB |
| Cores + factors, 4 streams | 274.841 ms | 262.583...284.000 | 4303.52 MiB |

Расчёт по чистым phase medians:

```text
forward + backward  370.320 ms
grad clip              8.190 ms
parallel optimizer   269.267 ms
grouped QR            49.286 ms
--------------------------------
estimated total      697.063 ms
```

Без QR: около 647.78 ms. Dense+Muon clean full-step control: около 394.5 ms.

Training peak:

```text
Dense + Muon                    9792.6 MiB
Tucker + parallel Muon + QR     9770.7 MiB
```

Разница всего 21.9 MiB в пользу Tucker. Full 10+30 JSON был снят на общей
карте с сильной переменной внешней нагрузкой; его full wall 839.34 ms нельзя
считать clean throughput. Для выбора Muon используется более стабильный
optimizer-only sweep.

Результаты лежат в:

```text
custom_backward/results/summary.json
custom_backward/results/autotune_a100.json
custom_backward/results/parallel_muon_autotune_a100.json
custom_backward/results/final_parallel_muon_mb1_s2_retract_w10_i30.json
```

## 10. Как адаптировать к другой Llama

### Шаг 1: сделай новый launcher

Скопируй существующий launcher, не редактируя validated 257M файл:

```bash
cp scripts/single_gpu/tucker_transformer/\
fineweb_standard_attention_muon_tucker_retract_1x_chinchilla.sh \
scripts/single_gpu/tucker_transformer/my_model_tucker.sh
```

Измени как минимум:

```text
N_LAYER
N_EMBD
N_HEAD
SEQ_LEN
MULTIPLE_OF
BATCH_SIZE
ACC_STEPS
EVAL_BATCH_SIZE
ITERATIONS/WARMUP
TARGET_PARAMETER_COUNT/TOLERANCE
rank policy
EXPERIMENT_NAME
```

Если нужна фиксированная FFN width, передай `--ffn-hidden-size`. Иначе Llama
вычисляет её из hidden size и `multiple_of`.

Сохрани обязательные flags:

```bash
--model llama
--linear-parameterization tucker
--tucker-forward-mode chunked_contract
--tucker-dense-adamw-matrices
--no-tucker-equal-params        # если нужен чистый rank-only Tucker
--tucker-retract-every-step
--dtype bfloat16
--opt muon
```

### Шаг 2: выбери Tucker modes/ranks

Feature dimension раскладывается функцией `balanced_factor_pair` на ближайшую
к квадрату точную пару множителей. Например:

```text
1024 -> 32 x 32
2816 -> 44 x 64
```

Доступны:

- `--tucker-rank auto`;
- один scalar `--tucker-rank N`, который clamp-ится к каждому mode;
- `--tucker-ranks r1,r2,r3,r4` для всех слоёв;
- отдельные `--tucker-attention-ranks`, `--tucker-gate-up-ranks`,
  `--tucker-down-ranks`;
- JSON `--tucker-rank-plan`, где есть запись для каждого заменяемого Linear.

Для новой модели предпочтителен per-family или per-module rank plan. Scalar
rank 259 был удобен именно потому, что после clamp полностью заполнял modes
32/44/64 текущей модели.

В pure Tucker режиме `--target-parameter-count` только проверяет результат.
Практический workflow:

1. временно убери target validation или поставь target `0`;
2. создай model и прочитай напечатанный `total parameters` и resolved plans;
3. выбери ranks;
4. зафиксируй новый target и разумную tolerance в launcher;
5. добавь runtime assertion в benchmarks.

Если требуется ровно dense parameter count, можно использовать
`--tucker-equal-params`; он добавляет trainable sparse residual. Это уже другой
архитектурный контракт, поэтому сравнение нужно явно пометить. Текущий target
использует `--no-tucker-equal-params` и residual=0.

### Шаг 3: проверь fast-kernel eligibility

Triton mode-pair kernels поддерживают только mode/factor dimensions <= 64.
Если хотя бы одна соответствующая dimension >64, код корректно уйдёт в
PyTorch/GEMM fallback, но скорость изменится.

Особенно внимательно проверяй prime/плохо факторизуемые hidden dimensions:
`balanced_factor_pair` может вернуть `1 x N`, и `N>64` отключит fused path.

Нельзя переносить A100 timing на новую shape family без замера.

### Шаг 4: выбери chunk size

Для текущего `B=16, S=1024` один chunk равен 16,384 tokens. Один chunk быстрее
и позволяет custom backward не создавать FP32 accumulators. Для новой модели:

1. начни с `chunk_size = batch_per_microstep * sequence_length`, если память
   позволяет;
2. если OOM, уменьши chunk в 2 раза;
3. сравни `recast` и `persistent`;
4. помни, что gradient accumulation не меняет tokens одного microstep.

### Шаг 5: обобщи hybrid cache

Не оставляй `out_features == 2816` для другой модели. Возможные решения:

- attach к каждому `TuckerLinear` boolean `persistent_work_cache` из model
  construction plan;
- выбрать top-K modules по `core_matrix.numel()` в рамках memory budget;
- определить family по полному module name (`gate_proj`, `up_proj`) вместо
  конкретного output size.

После изменения снова измерь peak всего training step, а не только cache bytes.

### Шаг 6: переавтотюнь Triton

`autotune_kernels.py` сейчас содержит только две production mode families.
Для другой модели измени `payload["shapes"]`, чтобы он перебирал все уникальные
`(n1,n2,r1,r2)` и `(m1,m2,r3,r4)` из resolved Tucker plan.

Запуск:

```bash
CUDA_VISIBLE_DEVICES=<GPU_UUID> python \
  experiments/fused_persistent_tucker/custom_backward/autotune_kernels.py \
  --tokens <CHUNK_SIZE> \
  --output experiments/fused_persistent_tucker/custom_backward/results/\
autotune_<model>_<gpu>.json
```

Перенеси selected `num_warps/num_stages` в `kernels.py` либо сделай dispatch
table по shape и GPU capability.

### Шаг 7: переавтотюнь Muon streams

`autotune_parallel_muon.py` сейчас hardcoded под 257M `make_config` и exact
parameter assertion. Для новой модели:

1. параметризуй config/model factory;
2. замени expected parameter count;
3. сохрани Dense-head assertion;
4. проверь candidates `microbatch in 1,2,4,8` и `streams in 1,2,4`;
5. выбирай по median и p10/p90, не по одному sample;
6. сравни phase peak memory.

На H100 обязательно повтори sweep: два и четыре streams на A100 отличались
примерно на 2%, поэтому A100 winner не является универсальным.

### Шаг 8: обнови тесты и benchmarks

Текущие `test_correctness.py`, `benchmark_muon.py` и
`autotune_parallel_muon.py` содержат production shapes/counts 257M. Для новой
модели они должны получать shapes автоматически из model или отдельного plan.

Минимальный acceptance checklist:

```text
[ ] lm_head имеет тип nn.Linear
[ ] число Tucker modules совпадает с ожидаемым
[ ] parameter count и residual count совпадают с контрактом
[ ] ни один Tucker module не использует materialize forward
[ ] forward parity с reference для каждой уникальной shape family
[ ] loss parity
[ ] dX и dCore/dU1..dU4 finite и close к reference
[ ] non-contiguous input
[ ] single-chunk и multi-chunk
[ ] gradient accumulation
[ ] cache invalidation после optimizer step
[ ] parallel Muon multi-step parity
[ ] Dense-head AdamW moments parity
[ ] optimizer checkpoint save/load/resume
[ ] grouped QR output invariance и orthogonality
[ ] несколько реальных training steps, loss finite/decreasing
[ ] 10+ warmup и 30+ measured на свободной GPU
```

## 11. Как адаптировать не к Llama

Сейчас `get_model` и `replace_all_linears_with_tucker` намеренно запрещают
`model != llama`. Для другой архитектуры недостаточно снять эту проверку.

Нужно явно решить:

1. Какие Linear заменять: attention projections, MLP projections или другие.
2. Как называется output head и должен ли он остаться Dense.
3. Есть ли tied weights между embedding и head. Нельзя заменить только одну
   сторону tied weight без изменения архитектуры.
4. Какие module suffix используются для attention/gate/up/down rank policies.
5. Как optimizer делит параметры между Muon и AdamW.
6. Совместим ли model forward/loss с Dense Liger linear cross entropy.
7. Какие dimensions factorable в modes <=64.

Рекомендуемое изменение: вместо recursive «заменить все `nn.Linear`» сделать
adapter с явными функциями:

```python
def should_tuckerize(full_name, module, model_config) -> bool: ...
def should_keep_dense(full_name, module, model_config) -> bool: ...
def rank_for(full_name, in_features, out_features) -> tuple[int, int, int, int]: ...
```

Для head нельзя полагаться только на точное имя `lm_head`: у другой модели это
может быть `output_projection`, `embed_out`, `language_model_head` и т.п.

Parallel Muon также использует names `.core_matrix`, `.U1`...`.U4`. Если новый
Tucker module сохраняет эти имена, optimizer path переносится. Иначе сделай
маркировку параметров по module identity, а не по строкам.

Текущая production схема single-GPU/DDP-friendly: в каждом DDP process model
на одном устройстве. Single-process model parallel/FSDP/TP требует отдельного
аудита stream ownership, sharded optimizer state и grouped QR.

## 12. Что считать честным сравнением

Dense и Tucker должны совпадать по:

- dataset/tokenizer;
- batch per microstep и gradient accumulation;
- sequence length;
- Dense `lm_head` и vocab;
- BF16 autocast;
- loss implementation;
- optimizer hyperparameters;
- clipping, scheduler и retraction policy;
- warmup/measurement count;
- GPU model/power state;
- отсутствию чужой GPU load.

Отдельно сообщай:

```text
forward
backward
forward+backward
clip
optimizer
retraction
full end-to-end step
peak forward/backward memory
peak full-training memory after optimizer-state creation
model/gradient/optimizer/cache bytes
```

Не складывай p10 одной фазы с median другой как «измеренный full step».
Допустимо дать такую сумму только как явно отмеченную estimate.

## 13. Известные ограничения и риски

- Full 39,250-step convergence run этой оптимизированной версии ещё не
  завершён. Correctness и короткая training dynamics проверены, но финальная
  quality должна быть подтверждена длинным run.
- Batching/foreach меняют floating-point reduction order; bitwise identity для
  Muon parameters не обещается. Three-step parity прошёл с `rtol=7e-4`,
  `atol=7e-5`; AdamW path совпал bitwise в тесте.
- Custom Triton fast path ограничен dimensions <=64; есть корректный fallback.
- `hybrid_gate_up` hardcoded под output width 2816.
- Kernel launch configs tuned на A100 PCIe, tokens=16384.
- Autotune/benchmark model factories hardcoded под 257M и требуют
  параметризации для других моделей.
- Grouped QR fast path не транспортирует optimizer state; при vector transport
  используется reference.
- Текущий training launcher hardcodes `NGPUS=1`. Для DDP сделай
  `NGPUS=${NGPUS:-1}` и проверь один GPU на process. TP/FSDP не валидированы.
- CUDA Graph не выбран: capture блокировался в стороннем backward/CUB path.
- Общий сервер часто сильно загружен; memory numbers валидны, wall time может
  быть загрязнён.

## 14. Checkpoint/resume

Architecture metadata и resolved Tucker plans сохраняются в checkpoint.
Resume должен использовать тот же:

```text
n_layer/n_embd/n_head/ffn/vocab
rank plan
Dense-head policy
equal-params policy
forward mode
optimizer split
```

`ParallelGroupedMuonLite.state_dict()` записывает Tucker routing как обычный
vanilla Muon (`use_muon=2`) для совместимости. После load parallel optimizer
восстанавливает transient route (`use_muon=4`), предотвращая двойной update.

Перед длинным run обязательно сделай реальный цикл:

```text
step -> save -> destroy process -> load -> step
```

и сравни loss/parameter delta с uninterrupted двумя steps.

## 15. Definition of done для новой модели/GPU

Адаптация считается законченной только когда:

1. Все пункты acceptance checklist пройдены.
2. Есть JSON с каждым raw timing sample и memory breakdown.
3. Есть clean Dense control с тем же Dense head.
4. Есть clean Tucker reference и optimized Tucker.
5. Выбранный cache policy не скрывает memory regression optimizer/retraction.
6. Выбранный stream count устойчив по p10/p90.
7. Выполнен checkpoint restart test.
8. Запущен хотя бы короткий реальный dataset run без NaN/Inf.
9. Для production quality завершён длинный convergence run или явно отмечено,
   что он ещё идёт.

## 16. Финальная команда для текущей 257M модели

```bash
cd <PROJECT_ROOT>
CUDA_VISIBLE_DEVICES=<GPU_UUID> \
TUCKER_CUSTOM_CACHE_POLICY=hybrid_gate_up \
TUCKER_PARALLEL_MUON=1 \
TUCKER_MUON_CORE_MICROBATCH=1 \
TUCKER_MUON_STREAMS=2 \
TUCKER_GROUPED_RETRACTION=1 \
TUCKER_ONLINE_CE=0 \
bash run_fused_dense_head_tucker.sh
```

На новой модели сначала используй `recast`, пока не обобщён hybrid plan:

```bash
TUCKER_CUSTOM_CACHE_POLICY=recast \
bash <NEW_MODEL_LAUNCHER>.sh
```

Главный принцип handoff: сначала сохранить архитектурный и численный контракт,
затем оптимизировать конкретные shape families. Не переносить hardcoded 257M
эвристики и A100 timing на новую модель как универсальные.
