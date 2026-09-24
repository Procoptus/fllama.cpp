# Применимые оптимизации: что править, где, как

Порядок = приоритет (влияние на наш сетап / стоимость). Каждый пункт: почему бьёт
в нас, что портить, что делать. Номера эффектов - из сообщений коммитов форка
(замеры автора, как правило на Strix Halo/gfx1151 + Vulkan, если не сказано иное);
без цифры помечено "(оценка, не измерено)".

## Приоритет 1. CL-a657590ae - true sparse gather для QSA attention (qwen4exp)

Ссылка: https://github.com/fewtarius/CachyLLama/commit/a657590aecf40532fd8fdd6e111786ee8614270c

### Почему это бьёт именно в наш случай

Наша модель qwen4exp использует QSA (Lightning Indexer -> `ggml_top_k`). В нашем
дереве `build_attn_qsa` ([src/models/qwen4exp.cpp:695](src/models/qwen4exp.cpp:695))
делает dense-маску: attention проходит по всем n_kv строкам KV-кэша, а немесе
обнуляются. На ctx 90000 это O(n_kv) чтения KV на каждый токен декода.
Коммит из форка заменяет это на gather: `ggml_get_rows` собирает только n_sel
строк, и `ggml_flash_attn_ext` считается по [d, 1, n_sel, nt].

Эффект по сообщению автора: для 131K-контекста (n_kv ~ 100K, indexer_top_k=2048,
compress_ratio=4 -> n_sel=2304) сокращение работы attention ~40x. Для ctx 90000
пропорция похожего порядка; на CUDA эффект по скорости не измерился автором -
направление +TG, величина "(оценка)".

Связка с нашим деревом: у нас уже есть собственный патч "QSA top_k-skip"
(ahead-24). Gather его не дублирует: top_k-skip пропускает indexer на мелких
батчах, gather сокращает сам attention на декоде. Условие активации gather из
сообщения: FA включён, нет alibi, decode-батч (n_tokens<=16),
`n_sel > pad256(width) < n_kv`. У нас fa=auto работает, decode одиночный -
условия выполняются. Порог по env `LLAMA_QSA_GATHER` (default 16384).

### Что портить

- `src/models/qwen4exp.cpp`: функции `qsa_gather_n_sel()`, `build_qsa_top_k()`
  (доработка существующей), `build_attn_qsa_gather()`; early-return в
  `build_attn_qsa()`
- `src/models/models.h`: объявления
- Против нашего дерева (уточнение 2026-09-04): gather-ОПЕРАЦИИ для KV у нас нет
  (ни `ggml_gather`, ни `GGML_OP_GATHER`); после `ggml_top_k`
  ([qwen4exp.cpp:684](src/models/qwen4exp.cpp:684)) attention считается плотной
  маской по всем n_kv в `build_attn_qsa`
  ([qwen4exp.cpp:695](src/models/qwen4exp.cpp:695)) - редукции чтения KV из
  top_k-выборки нет, именно её и добавляет коммит. Слово gather в нашем
  qwen4exp.cpp встречается только в комментариях (:110, :169, :171, :609, :1037,
  :1144, :1197). Имена/сигнатуры сверять вручную, конфликт с нашим top_k-skip
  минимален.

### Что делать

Портировать diff коммита (`git show` в read-only clone) поверх нашего
qwen4exp.cpp; прогнать `-c 90000` decode-бенччмарк до/после (tg). Проверить,
что при `LLAMA_QSA_GATHER=0` поведение идентично нынешнему (dense fallback).

## Приоритет 2. CL-f658fc5af - пропуск prompt-cache round-trip при n_parallel <= 1

Ссылка: https://github.com/fewtarius/CachyLLama/commit/f658fc5af98f404789b6200160892b9ebc87b8e4

### Почему бьёт

Наш запуск: `parallel=1` + `--cache-ram 3000`. В
[tools/server/server-context.cpp:1625-1646](tools/server/server-context.cpp:1625)
блок prompt_save/prompt_load/prompt_cache->update() выполняется на каждом
get_available_slot() без guard'а (проверено). При одной сслоте сохранённая
запись немедленно consumes следующим load - round-trip VRAM<->RAM вхолостую.
По сообщению автора на 128K/f16 KV: ~1 секунда за ход и ~22 GiB трафика
VRAM<->RAM за 40 минут нулевой пользы. Направление: -latency, -VRAM/RAM трафик.

### Что портить

Guard `n_parallel <= 1` вокруг блока save+load+update в get_available_slot() +
startup-warning при cache-ram > 0 с одной сслотой. Наш блок структурно тот же
(upstream-ный), diff ложится почти один-в-один (fork правит ту же upstream-коду).

### Что делать

Быстрая проверка без кода: `--cache-ram 0` - конфиг-эквивалент той же экономии
(кэш не создаётся вовсе). Если cache-ram не нужен при parallel=1 - достаточно
изменить строку запуска. Если хотим сохранить кэш для будущих parallel>1 -
портировать guard.

## Приоритет 3. CL-b83d23022 - seq_rm recurrent memory: fall-through вместо return false

Ссылка: https://github.com/fewtarius/CachyLLama/commit/b83d23022d1252c1c005e95e44db2cdc9fd154f8

### Почему бьёт

Наша модель - recurrent-hybrid (gated delta-net). В нашем дереве
[src/llama-memory-recurrent.cpp:161](src/llama-memory-recurrent.cpp:161) путь
частичного rollback заканчивается `return false;` (баг жив, проверено).
При streaming + tools + истории вызов common_context_seq_rm получает false ->
GGML_ABORT -> падение сервера. Это не скорость, а стабильность: мы гоняем
tools-агентские сессии на этой модели.

### Что портить

Заменить `return false` на fall-through в цикл очистки ячеек (1-2 строки).
Комментарий-обоснование в сообщении коммита.

### Что делать

Портировать; прогнать существующий `tests/test-recurrent-state-rollback.cpp`
(он уже в дереве - новый тест не нужен).

## Приоритет 4. CL-67f82f589 - резать backend splits по константе, не по выросшей ёмкости

Ссылка: https://github.com/fewtarius/CachyLLama/commit/67f82f589f8358104885d4815ca3effa764b7540

### Почему бьёт

Upstream #22789 сделал массив входов split'ов growable и заодно переключил
критерий разреза на `split->inputs_capacity`, которая только растёт. В нашем
дереве это есть: [ggml/src/ggml-backend.cpp:1341](ggml/src/ggml-backend.cpp:1341)
`if (split->n_inputs >= split->inputs_capacity)`. Мы держим experts и PLE на CPU
(`-ot "(.ffn_.*exp|per_layer_token_embd)=CPU"`), т.е. каждый граф режется
между CUDA и CPU; ratcheted cut point поднимает пик compute buffer
(n_copies раз при pipeline-параллелизме) и удлиняет splits. Направление:
-VRAM (пик compute buffer), +TG (меньше materialized копий). Величина "(оценка, не измерено)"
на CUDA; автор-оригинал (nathanw1014) мерил на Vulkan/Strix.

### Что портить

`ggml/src/ggml-backend.cpp`: сравнение с `GGML_SCHED_MAX_SPLIT_INPUTS` вместо
`inputs_capacity`; growable-массив оставить (он лечит assert >30 входов).
Минимальный правочный diff (cherry-pick из nathanw1014 76ad2ba9f в форке).

### Что делать

Портировать, проверить `-ot` CPU-оффлоад сценарий: пик CUDA compute buffer
в логе, отсутствие регрессии в `test-backend-ops`.

## Приоритет 5. CL-024dff234 - evict чекпойнт с наибольшим pos_min, а не insertion-oldest

Ссылка: https://github.com/fewtarius/CachyLLama/commit/024dff234d40871771926c6b152b0fa8d19d9dfe

### Почему бьёт

Мы запускаем `-ctxcp 3` на hybrid-модели (GDN: pos_min ограничен окном
рекурренции). Наш eviction сейчас - oldest `front()`
([tools/server/server-context.cpp:2317-2322](tools/server/server-context.cpp:2317),
проверено). Сценарий из сообщения: длинный первый ход создаёт mid-checkpoints
с низким pos_min; серия коротких ходов выталкивает их первыми; дальше любая
LCP-сессия не находит чекпойнт с `pos_min < pos_min_thold`
([server-context.cpp:3270](tools/server/server-context.cpp:3270)), do_reset
срабатывает и полный репроцесс промпта. Число подтверждено дословной цитатой из
сообщения коммита (проверено `git log -1 --format=%B 024dff234` 2026-09-04):
"Observed on task #270: full reprocess of 28123 tokens." Это наблюдение автора
на его модели (Qwen3.6-27B, гибрид DeltaNet), не замер на нашей сборке;
+PP направление - экономия prefill.

### Что портить

Логику выбора жертвы eviction из `std::list<common_prompt_checkpoint>`
([tools/server/server-task.h:569](tools/server/server-task.h:569)): искать max
pos_min вместо front(). ВНИМАНИЕ: форк-реализация частично привязана к
deferred_create_final_checkpoint (в нашем дереве нет) - портировать только
выбор жертвы, не всю форк-обвязку.

### Что делать

Адаптировать вручную ~10-20 строк. Тест: длинный ход, 7 коротких, затем ход с
ранним LCP - в логе не должно быть do_reset/full reprocess.

## Приоритет 6. CL-6a0db500c - SSD-backed KV cache (PORTED 2026-09-11)

Ссылка: https://github.com/fewtarius/CachyLLama/commit/6a0db500ca1058e06a232c02c50eb5df56b0d151

СТАТУС 2026-09-11: перенесено в ветку `ssd-kv-cache` (net-diff подсистемы:
kv-ssd-cache.{h,cpp}, server-context-page-manager, system-prompt-cache,
интеграция в server-context, llama-context, hparams/args + 13 флагов
`--cache-ssd*`). POSIX-адаптация под MSVC: mmap -> CreateFileMapping/
MapViewOfFile, madvise -> no-op, host-RAM -> GlobalMemoryStatusEx, mkdir ->
_mkdir. llama-server.exe собирается под MSVC; тест test-kv-ssd-user-isolation
проходит. 48 BLOCKED-фиксов SSD-группы теперь переносимы (ещё не перенесены).

### Почему бьёт

ctx 90000 + частые перезапуски/длинные агентские сессии: cold-start reprocess
всего промпта дорог. Подсистема (kv_page_manager 'KVPG', hot/warm/cold RAM+SSD
уровни, LCP-match восстановление поверх hybrid checkpoint) даёт -latency на
restart. За ней стоят 40+ фиксов из BLOCKED (см. 01-registry, раздел 3) -
портить надо базой, иначе фиксы некуда класть.

### Что делать

Отдельный большой проект. Проверить до начала: POSIX-зависимости
(fsync/fadvise/пути) и то, что подсистема не ломает наш kv-unified + GDN путь.
Решение "портить/не портить" принимать после того, как приоритеты 1-5 закрыты:
stability-фиксы и QSA gather дают эффект без этой махины.

## Приоритет 7. CL-56dca0825 + CL-cc60f8912 - global system prompt KV cache (PORTED 2026-09-11)

Ссылки: https://github.com/fewtarius/CachyLLama/commit/56dca0825e1d0b3a4b5f00a1fc1e59f2c6a900f8 ,
https://github.com/fewtarius/CachyLLama/commit/cc60f89123cc5c800717bed1b9c5892d7aac12c4

СТАТУС 2026-09-11: перенесено вместе с базой (ветка `ssd-kv-cache`), флаги
`--cache-ssd-system-prompts` / `--cache-ssd-system-max-days`.

Глобальный кэш KV для общего system prompt (переживает reset сслоты; warm slots
после do_reset падают в него). Часть SSD-подсистемы (compat hash с page_manager).
Эффект: -latency на first-turn при одном big system prompt у многих сессий.
Бьёт слабее приоритета 6-го и только при нашем профиле "много коротких сессий
с одним системником". Портить только вместе/после базы.

## Приоритет 8. CL-022c16ed5 - защитный guard на K-shift NULL buffer

Ссылка: https://github.com/fewtarius/CachyLLama/commit/022c16ed51f271ce016f6be5714fe23924c24c06

В нашем дереве [src/llama-kv-cache.cpp:1526](src/llama-kv-cache.cpp:1526)
set_input_k_shift - GGML_ASSERT(host buffer) без null-guard (проверено).
Форк-сценарий краха (ISWA + kv-unified без RoPE-слоёв) на нашей модели
невыполним: у qwen4exp есть RoPE-слои (IMROPE), буфер аллоцируется. Портить
стоит только defensive-часть (early-return + null-guard по образцу
llm_graph_input_attn_kv). conv_hash-часть коммита - BLOCKED (привязана к
SSD/sys-cache). Приоритет низкий: страховка от будущего изменения графа.
(оценка, не измерено)

## Приоритет 9. CL-dd3fccf1a - CONFIG-подтверждение: не квантовать indexer/KV при lightning-indexer

Ссылка: https://github.com/fewtarius/CachyLLama/commit/dd3fccf1a95b3f91c140ca9859c9f94f1c402ca4

Коммит pin'ит DSA indexer key cache в f16 под квантованным `-ctk`, т.к. fused
indexer-ядра читают только f16, а деградация на квантованном пути измерена:
0.8ms -> 97ms на dispatch к 12k контекста (Vulkan). Сам код - про DeepSeek DSA
и Vulkan-ядра, нам не переносить. Но правило конфигурации для нашего QSA
пути то же и оно УЖЕ выполнено: `-ctk f16 -ctv f16`. Действие: не включать
квантованный KV cache без проверки, не потеряли ли indexer fast-path.

## Приоритет 10. Знание из CL-5044107be - ISA-автоопределение (только идея)

Ссылка: https://github.com/fewtarius/CachyLLama/commit/5044107becc80af940386cf6f5234abd4955816e

Форк добавил bash-детект /proc/cpuinfo, потому что сборка с GGML_NATIVE=OFF
теряла AVX-512 пути: -30-100% на CPU-оффлоад слоях (замер автора, Zen4).
Для нас код неприменим (Linux), но проблема та же: наш `-ot ...=CPU` оффлоад
чувствителен к ISA-флагам. Мы уже задаём AVX2/AVX512/VNNI вручную - сверку
проводили в [06-build-flags-audit.md](06-build-flags-audit.md). Действие:
ничего, держать флаги явными.

## Не применять

- CL-757361a70 / CL-54e51d11f / CL-5dd48c54b - реальные qwen4exp/hybrid фиксы,
  но привязаны к отсутствующему API `seq_rm_attn_only` и SSD path (BLOCKED).
  Если когда-нибудь портим SSD-базу - брать их первыми, т.к. именно qwen4exp.
- CL-376bcc14d (MTP draft head для qwen4exp) - MTP у нас не включён; если
  включим speculative - пересмотреть.
