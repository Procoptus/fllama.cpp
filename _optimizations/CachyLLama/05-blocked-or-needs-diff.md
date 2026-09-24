# 05-blocked-or-needs-diff.md - BLOCKED-группы: чем заблокировано, объём, зависимости

Файл раскрывает строки BLOCKED из [01-registry.md](01-registry.md) (раздел 3, 48
коммитов). Строки в реестре остаются заглушками с указанием блокирующей базы;
детали - здесь. Числа объёмов получены `git show --stat` в read-only клоне
`D:\dev\llama_compiling\_forks_opt\cachyllama` (2026-09-04).

## BLOCKED: CL-6a0db500c (+ CL-56dca0825, CL-cc60f8912) - SSD-backed KV cache подсистема

База: [CL-6a0db500c](https://github.com/fewtarius/CachyLLama/commit/6a0db500ca1058e06a232c02c50eb5df56b0d151) -
SSD KV cache (kv_page_manager 'KVPG', hot/warm/cold уровни RAM+SSD,
восстановление поверх hybrid checkpoint). Спутники той же подсистемы:
[CL-56dca0825](https://github.com/fewtarius/CachyLLama/commit/56dca0825e1d0b3a4b5f00a1fc1e59f2c6a900f8)
(global system prompt KV cache) и
[CL-cc60f8912](https://github.com/fewtarius/CachyLLama/commit/cc60f89123cc5c800717bed1b9c5892d7aac12c4)
(wiring в жизненный цикл сервера).

### Объём diff (git show --stat, проверено)

- CL-6a0db500c: 28 файлов, +5335/-1357. Ключевые новые файлы:
  `common/kv-ssd-cache.cpp/.h`, `tools/server/server-context-ssd-cache.cpp/.h`,
  правки `tools/server/server-context.cpp`, `tools/server/server-task.*`.
- CL-56dca0825: 3 файла, +801 (`common/kv-ssd-system-cache.cpp` 601 строк,
  `.h` 198).
- CL-cc60f8912: 3 файла, +304 (`tools/server/server-context.cpp` +288).
- Итого минимум ~6400 строк нового кода до учёта фиксов.

### POSIX-блокировки (подтверждены в диффе базы)

- `#include <unistd.h>` - 3 входа в диффе CL-6a0db500c.
- `fsync(fd)` - 2 вызова (флаг `--cache-ssd-no-fsync` позже появился как
  отдельный коммит CL-a4e459ef0).
- CL-43d781ac0 использует kernel readahead (posix_fadvise-механика) - аналога на
  Windows нет.
- Windows-эквиваленты возможны (`_commit` / `FlushFileBuffers`,
  `ReadFile`+`FILE_FLAG_SEQUENTIAL_SCAN`), но это не косметическая правка: всю
  file-io часть надо переписывать и тестовать под NTFS.

### Зависимости

За базой стоят оставшиеся ~45 BLOCKED-фиксов реестра: SSD cache (~15),
SSD cold-start (~7), sys-prompt cache (~7), checkpoint ring (~9),
prompt/sys cache metadata. Без базы фиксы переносить некуда. Дополнительно 3
коммита (CL-9b06c5a95 и др.) требуют ещё и API `seq_rm_attn_only` (см. ниже) -
двойная блокировка.

### Вердикт

HIGH-COST. Для нашего сетапа (ctx 90000, одна сслота, Windows) практический
смысл - только -latency на холодный старт/перезапуск. Ожидаемый эффект автором
в цифрах не заявлен (оценка, не измерено). Портировать целиком под Linux-API
массив на Windows - неоправданный риск; решение принимать только при реальной
боли от cold-start reprocess.

## BLOCKED: API seq_rm_attn_only (CL-54e51d11f, CL-5dd48c54b, CL-757361a70)

- В нашем дереве API `seq_rm_attn_only` отсутствует: по seq_rm у нас одна
  функция [src/llama-memory-recurrent.cpp:161](src/llama-memory-recurrent.cpp:161)
  без разделения attention/recurrent-доменов.
- Это самые qwen4exp-специфичные фиксы форка (индексерный кэш синхронно
  attention под seq_rm_attn_only; "Invalid input batch" на multi-turn, issue
  #8). При этом наш прямой баг seq_rm (неудачный rollback -> return false
  вместо fall-through) чинится отдельно и дёшево: CL-b83d23022, Приоритет 3 в
  [02-applicable.md](02-applicable.md).
- Брать только если когда-нибудь появится need в семантике selective remove по
  attention-домену.

## NO-DIFF: CL-f629077d1 - re-enable ISWA chunk reuse + prompt cache current

[CL-f629077d1](https://github.com/fewtarius/CachyLLama/commit/f629077d1337a4ed8b502ce7a42010cdcbb77705)
"server: keep prompt cache current + re-enable ISWA chunk reuse" (3 файла,
+20/-6). Переклассифицирован из BLOCKED в N/A 2026-09-04 (раздел 11 реестра,
proof в [04-not-applicable.md](04-not-applicable.md), раздел 8). Ответ на
вопрос "какой класс KV-памяти у qwen4exp при kvu=true":

- `llama_memory_hybrid_idx` ([src/llama-model.cpp:2509-2529](src/llama-model.cpp:2509),
  unified = cparams.kv_unified на :2526; needs_mem_idx для qwen4exp :2462).
- Внутри - обычный `llama_kv_cache` + `llama_memory_recurrent`
  ([src/llama-memory-hybrid.h:89-90](src/llama-memory-hybrid.h:89)).
- `llama_kv_cache_iswa` для qwen4exp не инстанцируется никогда; правимый
  коммитом `llama_kv_cache_iswa::get_can_shift()`
  ([src/llama-kv-cache-iswa.cpp:253-257](src/llama-kv-cache-iswa.cpp:253)) -
  недостижимый для нас путь.
- Фактический K-shift gate - `llama_kv_cache::get_can_shift()`
  ([src/llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189)): false для
  mrope n_pos_per_embd=4; ctx_shift штатно отключён
  ([common/common.cpp:1457](common/common.cpp:1457)).
- Server-часть коммита привязана к fork-only prompt-cache подсистеме - без базы
  переносить некуда.

## Ограничение метода

- Объёмы diff - по `git show --stat` одиночных коммитов; кумулятивная площадь
  группы с учётом взаимных переписываний может быть меньше (повторные правки
  одних файлов).
- POSIX-блокировки проверены grep'ом по диффу базы; полный аудит portability
  (путь-разделители, case-insensitive FS, блокировки файлов на Windows) при
  реальном портировании обязателен.
- Разделение "фикс зависит от базы" принято по наличию символов базы в диффе
  коммита (page_manager, checkpoint ring API, seq_rm_attn_only); фиксы,
  случайно задетые теми же файлами, в BLOCKED не записаны.
