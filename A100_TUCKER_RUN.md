# Tucker + Dense lm_head: запуск на A100

Финальная реализация находится в
`experiments/fused_persistent_tucker/custom_backward/`.

```bash
cd /home/rustikkabirov/projects/tucker_fused_a100_20260825
CUDA_VISIBLE_DEVICES=<A100_UUID> \
PYTHON_BIN=.venv-a100/bin/python \
bash experiments/fused_persistent_tucker/custom_backward/run_final.sh
```

Она фиксирует ровно 84 внутренних Tucker-модуля, 257,676,352 обучаемых
параметра и Dense `nn.Linear(1024,50304)` head. Online Tucker CE и
материализация внутренних Dense-весов отключены.

Итоговый A100 F+B: 370.320 ms / 8343.4 MiB в default `hybrid_gate_up` и
370.897 ms / 8211.0 MiB в minimum-memory `recast`. Подробная методика, parity,
Muon memory и traces описаны в `custom_backward/README.md`.

Default optimizer теперь использует полноценный parallel Muon: 84 Tucker-core
матрицы распределяются между двумя CUDA streams, 336 факторов группируются, а
AdamW для Dense `lm_head` перекрывается с ними. A100 same-process sweep выбрал
`core_microbatch=1`, `streams=2`: 269.267 ms против 554.884 ms у
последовательного reference, без роста optimizer-phase memory. Настройки уже
включены в `run_final.sh`.

Полный документ для передачи проекта другому Codex и адаптации под другие
размеры Llama, H100 и не-Llama архитектуры: `TUCKER_PARALLEL_HANDOFF.md`.
