# Применимые оптимизации: что править, где, как

Порядок = рекомендуемый порядок выполнения. Эффекты - инженерная оценка для
RTX 3080 sm_86 + i7-12700K (8P/16) + qwen4exp Q3_K_XL + ctx 90000. Числа из
описаний ik приведены в скобках там, где они были.

---

## Приоритет 1. #2225 - bucket top_k для `GGML_OP_TOP_K` на CPU

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/2225
- Область: CPU / TG / QSA индексер
- Эффект: **-20..40 % времени индексера** при n_kv=90000, то есть **+2..5 % TG**
  на длинном контексте (ik заявлял +3 % TG @128k)
- Сложность: средняя. Риск: средний

### Почему это бьёт именно в наш случай

`qwen4exp` использует QSA - разреженное внимание с отдельным индексером и третьим
кэшем индексера (`llama_memory_hybrid_idx`). Индексер вызывает `ggml_top_k` по
строкам длиной до `n_kv` = 90000 на каждый decode-ubatch.

Вызов в модели: `src/models/qwen4exp.cpp:531` (`build_qsa_top_k`), используется в
`build_attn_qsa` (`src/models/qwen4exp.cpp:695`).

GPU путь для этой операции у нас недоступен: guard `ne0 <= 1024` в
`ggml/src/ggml-cuda/ggml-cuda.cu:5360`, плюс CUB в сборке не найден. Значит
операция целиком исполняется на CPU и входит в критический путь TG.

Upstream до сих пор использует `std::partial_sort`:

- `ggml/src/ggml-cpu/ops.cpp:8557` - `ggml_compute_forward_top_k_f32`
- `ggml/src/ggml-cpu/ops.cpp:8583` - `std::partial_sort`
- `ggml/src/ggml-cpu/ops.cpp:8591` - трюк `std::swap(dst[0], dst[1])`

### Что портить

Алгоритм bucket-select (гистограмма по старшим разрядам float, затем доработка
выбранного бакета) за O(N) вместо O(N log K) от `partial_sort`.

Требования к переносу:
1. Не менять сигнатуры и контракт операции.
2. Сохранить детерминизм/стабильность порядка индексов при равных значениях.
3. Корректно обработать `-INFINITY` и NaN.
4. Не трогать ветку `n_threads > 1` без необходимости: автор ik оговаривал, что
   на batch-пути бакетный метод не быстрее `partial_sort`, выигрыш на одиночном
   decode-пути.

---

## Приоритет 2. #1599 - квантованный KV cache (`-ctk` / `-ctv`)

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1599
- Область: KV-cache / VRAM / FlashAttention
- Статус: механизм **уже в upstream**, нужен только параметр запуска
- Эффект: **-40..50 % VRAM на KV**, **+2..6 % PP/TG** косвенно (меньше memory
  traffic), риск деградации качества
- Сложность: нет (конфиг). Риск: средний (качество)

### Состояние

У вас `ctk=f16`, `ctv=f16` при `ctx-size=90000`. Это самый прожорливый вариант KV.

Механизм в upstream:
- `-ctk` - `common/arg.cpp:2434`
- `-ctv` - `common/arg.cpp:2447`

### Обязательное условие, которое у вас уже выполнено

При `K->type != V->type` Flash Attention полностью выключается:
`ggml/src/ggml-cuda/fattn.cu:447`. У вас `GGML_CUDA_FA_ALL_QUANTS=ON`
(`build_avx_512/CMakeCache.txt:674`), поэтому смешанные варианты доступны.

D=256 поддерживается: `ggml/src/ggml-cuda/fattn-vec.cuh:585`.
Деквант Q8_0/Q4_K: `ggml/src/ggml-cuda/fattn-common.cuh:641`.

### Что делать

1. `-ctk q8_0 -ctv q8_0` - безопасная ступень, почти без потери качества.
2. Затем по желанию `-ctk q8_0 -ctv q4_K` - больше VRAM, больше риск.
3. Замер TG/PP и обязательная субъективная/объективная проверка качества на
   вашем рабочем промпте.

---

## Приоритет 3. #1261 - self-speculative ngram

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1261
- Область: TG / speculative decoding
- Статус: **уже в upstream**
- Эффект: **+10..25 % TG** на шаблонном/повторяющемся тексте (chat-шаблон, код,
  thinking-блоки), около 0 на креативном
- Сложность: нет (конфиг). Риск: средний

### Почему только ngram

`qwen4exp` не имеет MTP/nextn: `llama_model_n_layer_nextn`
(`src/llama-model.cpp:2740`), а `DRAFT_MTP` требует эти слои. Доступные типы
через `--spec-type` (`common/arg.cpp:4249`):

- `ngram-simple`, `ngram-map-k`, `ngram-map-k4v`, `ngram-mod`, `ngram-cache`
  (`common/speculative.cpp:39`)

### Что делать

```
--spec-type ngram-map-k --spec-draft-n-max 3
```

`--spec-draft-n-max` - `common/arg.cpp:4140`. Подбирать 2..4 по замерам.

### Побочный эффект, важный для #1759

Включение speculation заполняет `n_rs_seq`:
`cparams.n_rs_seq = params.speculative.need_n_rs_seq()`
(`common/common.cpp:1724`, `common/common.h:394`). Без этого #1759 бессмыслен
(см. `05-blocked-or-needs-diff.md`).

Цена: `n_rows = mem_size * (1 + n_rs_seq)`
(`src/llama-memory-recurrent.cpp:101`) - рост потребления памяти, и
`split_equal(n_ubatch, unified, n_rs_seq + 1)`
(`src/llama-memory-recurrent.cpp:445`) - ограничение на батчинг.

---

## Приоритет 4. #1137 / #1403 - merged gate+up experts (только переконвертация GGUF)

- Ссылки: https://github.com/ikawrakow/ik_llama.cpp/pull/1137 ,
  https://github.com/ikawrakow/ik_llama.cpp/pull/1403
- Область: MoE / PP / VRAM
- Статус: **код уже в upstream**, но ваш GGUF может его не использовать
- Эффект: **+3..8 % PP** на CPU-оффлоаде экспертов (ik заявлял 6-10 %), один
  `mul_mat_id` вместо двух на слой
- Сложность: нет (переконвертация). Риск: нет

### Проверка

```
gguf_dump.exe Qwen3.8-Flash-Next-UD-Q3_K_XL-00001-of-00003.gguf | findstr ffn_gate_up_exps
```

Пусто - значит файл содержит раздельные `ffn_gate_exps` и `ffn_up_exps`, и
выигрыш в один `mul_mat_id` на слой теряется.

### Что в коде уже готово

- `create_tensor_gate_up_exps()` - `src/llama-model.cpp:3145` (фолбэк на
  раздельные тензоры, если merged нет)
- вызов в модели - `src/models/qwen4exp.cpp:251`
- enum тензора - `src/llama-arch.cpp:445` (`blk.%d.ffn_gate_up_exps`)
- граф: «one mul_mat_id, then split into gate and up views» -
  `src/llama-graph.cpp:2114`, `has_gate = gate_exps || gate_up_exps` -
  `src/llama-graph.cpp:2164`

### Как получить merged тензоры

Конвертер: `--fuse-gate-up-exps` (`convert_hf_to_gguf.py:152`,
`conversion/base.py:626`).

### Ограничение

Как **код** эти PR непереносимы: `merge_qkv` в дереве нет, дифф #1137 идёт по 16
файлам через `ggml.c`/loader/quantize. Используется только уже существующий
upstream-механизм.

---

## Приоритет 5. #1049 - сокращение синхронизаций бэкендов в sched

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1049
- Область: CUDA / CPU scheduling / TG+PP
- Эффект: **+1..3 % TG/PP** (ik чисел не публиковал)
- Сложность: средняя. Риск: средне-высокий

### Механизм ik

`std::array<bool, GGML_SCHED_MAX_BACKENDS> needs_sync{{true}}` в
`ggml_backend_sched_compute_splits`: синхронизация бэкенда пропускается, пока
`split->n_inputs == 0`, и снова ставится при появлении входов.

### Наше состояние

`ggml_backend_sched_compute_splits()` - `ggml/src/ggml-backend.cpp:1654`
синхронизирует безусловно на каждый вход - `ggml/src/ggml-backend.cpp:1687`.

При `-ot "(.ffn_.*exp|per_layer_token_embd)=CPU"` на каждый ubatch строится
гибридное CPU+CUDA расписание, и лишние `ggml_backend_synchronize` сериализуют
CPU<->GPU на TG.

### Почему риск выше обычного

В нашем дереве поверх этой же функции добавлена логика копирования только
активных экспертов (`used_ids` / `copy_experts`) -
`ggml/src/ggml-backend.cpp:1749`. Дифф ik ложится прямо на этот код с
конфликтами. Только ручной перенос.

Пропуск необходимой синхронизации даёт race-условие. Обязательна проверка
детерминизма вывода (одинаковый seed -> одинаковый текст) до и после.

---

## Приоритет 6. #860 - не форматировать имена тензоров без нужды

**Статус: APPLIED 2026-09-03** (ручной порт, локально, не коммитится)

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/860
- Область: CPU / build-overhead / TG
- Эффект: **~1 % TG**
- Сложность: низкая. Риск: низкий

Имена тензоров форматируются через `vsnprintf` при каждом построении графа -
`ggml/src/ggml.c:1957`. На decode граф строится на каждый ubatch. Идея переноса:
не форматировать, когда имена не нужны (нет лога, нет отладки).

Единственный мелкий портируемый остаток из «хвоста» инвентаря.

Что перенесено:
- `ggml_format_name_fast()` - static inline без `vsnprintf`, `ggml/src/ggml.c:1963`
- вызовы заменены в `view/reshape/permute/transpose/cont/copy` (13 точек)
- хот-путь `graph_get_cb` в `src/llama-context.cpp:2522` - ручной побайтовый
  `%s-%d` вместо `vsnprintf`
- ханки PR по `ggml_set_param`/`ggml_top_k` неприменимы: в нашем дереве этих
  вызовов уже нет

---

## Приоритет 7. #1427 - перепись CPU `rms_norm`

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1427
- Область: CPU / AVX-512
- Эффект: **~0 % при текущем оффлоаде**; +1..2 % PP если расширять CPU-часть
- Сложность: низкая-средняя. Риск: низкий

`ggml_compute_forward_rms_norm_f32()` - `ggml/src/ggml-cpu/ops.cpp:3924` до сих
пор содержит `// TODO: optimize` - `ggml/src/ggml-cpu/ops.cpp:3951`, скалярный
`sum += (ggml_float)(x[i00]*x[i00])`.

Важно: fusion `rms_norm+mul` у нас **уже есть** -
`ggml/src/ggml-cpu/ops.cpp:4009`, чего нет у ik. Встраиваться в существующий
шаблон `FUSE_OP`, не ломая его. CPU `can_fuse` разрешает только
`{RMS_NORM, MUL}` - `ggml/src/ggml-cpu/ggml-cpu.c:3081`.

Брать только если планируется больше слоёв на CPU: при текущем `-ot` norm-операторы
назначаются CUDA.

---

## Приоритет 8. #1417 - `top_n_sigma` без `pow()`

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1417
- Область: Sampling
- Эффект: **0 % у вас** (сэмплер выключен), 4000->8000 t/s в заявке ik это
  пропускная способность самого сэмплера, не TG
- Сложность: низкая. Риск: низкий

`llama_sampler_top_n_sigma_apply()` - `src/llama-sampling.cpp:3237`: три полных
прохода по vocab и `pow(cur_p->data[i].logit - mean, 2)` -
`src/llama-sampling.cpp:3263` -> `d*d`.

У вас `top_n_sigma` не задан (по умолчанию -1, `common/common.h:250`). Брать
только если начнёте использовать `--top-nsigma` (`common/arg.cpp:2045`).

---

## Приоритет 9. #1560 - CLI для резерва compute buffer

- Ссылка: https://github.com/ikawrakow/ik_llama.cpp/pull/1560
- Область: VRAM / стабильность
- Эффект: **0 % скорости**, -VRAM / предсказуемость
- Сложность: низкая. Риск: низкий. Статус: NO-DIFF (дифф не верифицирован)

Механизм резерва по худшему графу в upstream **уже есть**: `n_outputs_max` -
`src/llama-context.cpp:2086`, буферы `2*n_vocab*n_outputs_max`, резерв
`src/llama-context.cpp:622-675` и `src/llama-context.cpp:2414-2456`.

Остаток работы: в `common/arg.cpp` нет CLI-флага, чтобы задать это явно. Актуально
при ctx=90000 + 3 чекпоинтах.

---

## Не-PR рычаги конфигурации

### `-DGGML_AVX512_BF16` - НЕ применять (ошибка отменена 2026-09-02)

Ранее здесь стояла рекомендация включить эту опцию со ссылкой «12700K имеет AVX512-BF16». Это неверно, сборка с ней падает. Не применять. Детали, доказательства и верифицирующий тест: `06-build-flags-audit.md`, раздел «BF16: отмена рекомендации (2026-09-02)».

Важно: `FindSIMD.cmake:92` (`ggml/src/ggml-cpu/cmake/FindSIMD.cmake:92`)
автоопределяет только AVX / AVX2+FMA / AVX_VNNI / AVX512. VBMI, VNNI, BF16 нужно
указывать вручную - вы это делаете правильно.

### `-t` / `-tb` тюнинг

E-ядра отключены = 8 ядер / 16 потоков, стоит `threads=13`, `threads-batch=16`.
Для AVX-512 heavy-пути на P-ядрах hyper-threading часто вредит. Прогнать
`-t 8` / `-t 13` / `-t 16` на `llama-bench` (`-t` -
`tools/llama-bench/llama-bench.cpp:691`, `-ts` - `tools/llama-bench/llama-bench.cpp:945`).
`threads-batch=16` оставить.

### `-fa on` vs `auto`

FA по логам llama-server у вас уже работает. Явный `-fa on`
(`common/arg.cpp:1751`) нужен только чтобы зафиксировать поведение; отдельно
прогнать `-fa off` как baseline, чтобы не принимать желаемое за действительное.

### `-ncmoe` вместо ручного `-ot` regex (#2262)

Заменить `-ot "(.ffn_.*exp|per_layer_token_embd)=CPU"` на `-ncmoe`
(`common/arg.cpp:2789`) плюс отдельный `-ot` (`common/arg.cpp:2775`) только для
`per_layer_token_embd`.

Зачем: ручной `-ot` блокирует `--fit` - `common/fit.cpp:483` бросает
`common_params_fit_exception`, если `tensor_buft_overrides` уже задан. С `-ncmoe`
открывается MoE-aware автоподбор слоёв: `common/fit.cpp:547`
(`set_ngl_tensor_split_tbo`), `:575` (`LAYER_FRACTION_MOE`), `:618`
(`pattern_moe_all` -> CPU buft), `:729` (`global_surplus_cpu_moe`).

### `GGML_CUDA_GRAPH_OPT=1`

Уже задано пользователем в батнике. Это **runtime** переменная:
`ggml/src/ggml-cuda/ggml-cuda.cu:4441` читает `getenv`. В cmake передавать
бессмысленно (см. `06-build-flags-audit.md`).
