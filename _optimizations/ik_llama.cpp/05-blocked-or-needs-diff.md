# Заблокировано зависимостью или не верифицирован дифф

## BLOCKED: #1759 - асинхронные копии recurrent-state

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1759
- Область: TG / delta-net / qwen4exp
- Потенциал: +1..3 % TG
- Сложность: высокая. Риск: высокий

### Двойная блокировка

**Блокировка 1 - нет предпосылки.** Синхронное заполнение recurrent-state
находится здесь:

- `src/llama-graph.cpp:333` - `data[i] = mctx->get_recr()->s_copy(i)`
- `src/llama-graph.cpp:1101`
- `src/llama-graph.cpp:1145`
- `src/llama-graph.cpp:1219`
- `build_rs()` - `src/llama-graph.cpp:3408`

Но весь этот путь коротко замыкается:

```cpp
if (n_rs_seq == 0) return src0;
```

`src/llama-memory-recurrent.cpp:1310`

`n_rs_seq` ненулевой только при включённом speculation
(`cparams.n_rs_seq = params.speculative.need_n_rs_seq()`,
`common/common.cpp:1724`). Значит сначала нужен #1261.

**Блокировка 2 - нет API.** `llama_spec_ckpt_*` в дереве отсутствует (`git grep`
пуст). Дифф ik использует этот API.

### Цена включения предпосылки

- `n_rows = mem_size * (1 + n_rs_seq)` - `src/llama-memory-recurrent.cpp:101`
- snapshot plane - `src/llama-memory-recurrent.cpp:192`
- `split_equal(n_ubatch, unified, n_rs_seq + 1)` -
  `src/llama-memory-recurrent.cpp:445`
- две conv-строки, `n_slots = cparams.n_rs_seq + 1` -
  `src/models/qwen4exp.cpp:1130`, `:1163`

То есть рост памяти и ограничение на батчинг. Решение принимать только после
замеров #1261.

### Поддержка rollback в архитектуре есть

`llm_arch_supports_rs_rollback` включает `LLM_ARCH_QWEN4EXP` -
`src/llama-arch.cpp:1099`, `:1103`; клампинг в `src/llama-context.cpp:104`.
Тест существует: `tests/test-recurrent-state-rollback.cpp`.

---

## NO-DIFF: #1560 - резерв compute buffer по худшему графу

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1560
- Причина: локального клона ik_llama.cpp на диске нет, `git fetch` запрещён
  правилами read-only. Вердикт дан по описанию PR и по коду upstream.

Механизм в upstream найден: `n_outputs_max` - `src/llama-context.cpp:2086`,
буферы `2*n_vocab*n_outputs_max`, резерв `src/llama-context.cpp:622-675` и
`src/llama-context.cpp:2414-2456`. Остаток работы - только CLI-флаг в
`common/arg.cpp`.

Перед портированием получить дифф:
`gh pr diff 1560 --repo ikawrakow/ik_llama.cpp`

---

## NO-DIFF: #1137 / #1403 в части «как код»

Диффы не верифицированы по той же причине. Но для нас это не блокировка: как код
они непереносимы (см. `04-not-applicable.md`, группа 5), а используемая часть
(merged `ffn_gate_up_exps`) в upstream уже есть. Действие - проверить GGUF и при
необходимости переконвертировать (см. `02-applicable.md`, приоритет 4).

---

## Ограничение метода: усечённый инвентарь

Инвентарь main-ветки ik был усечён на секции C.5, поэтому раздел «топ-15» не
был выдан целиком и восстановлен по таблице кандидатов плюс вердиктам трёх
проверочных субагентов.

Что это означает на практике: в хвосте списка теоретически могли остаться
позиции, но все они попадают в уже закрытые категории (IQK, multi-GPU, чужие
архитектуры, sm_60/sm_70, ik-only ядра). Полнота по числу разобраных ID - 76.

Как закрыть пробел в следующий раз: сохранить полный вывод инвентаря в
`07-raw-inventory.md` (файл создан, но заполнен частично - см. его шапку).
