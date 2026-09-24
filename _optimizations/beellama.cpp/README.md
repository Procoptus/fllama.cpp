# Реестр проверенных оптимизаций из Anbeeld/beellama.cpp

См. также: [Сводный отчёт по обоим форкам](../SUMMARY.md)

Форк ориентирован на KVarN (низкобитный KV cache), TurboQuant/TCQ квантование,
DFlash/MTP спекулятивное декодирование и ROCm/HIP. Generic CUDA-работы (вне
turbo/kvarn/spec путей) в форке мало. Это принципиально отличается от сетапа
пользователя (Windows + CUDA + RTX 3080 + Q3_K_XL + KV f16), поэтому
значительная часть форка не применима.

Данные: форк HEAD `cd3c41e73`, clone `D:\dev\llama_compiling\_forks_opt\beellama`,
merge-base с upstream `95ef7fc16`, форк ahead ~848 коммитов (см. сырой инвентарь).
Рабочие материалы аудита (логи git, диффы): `D:\dev\llama_compiling\bee_audit\`.
Сырой инвентарь: [07-raw-inventory.md](07-raw-inventory.md).

## Целевой сетап

| Параметр | Значение |
|---|---|
| CPU | i7-12700K (E-ядра отключены, AVX-512 разблокирован, нет AVX512-BF16) |
| GPU | RTX 3080 10GB (sm_86 Ampere), CUDA backend |
| RAM | 64GB DDR4-3600 |
| ОС | Windows 11 25H2, MSVC/cl + Ninja Release |
| Сборка | CUDA arch 86, BUILD_SHARED_LIBS=OFF, GGML_NATIVE=ON, LTO=ON, CCACHE=ON, OPENMP=ON, GGML_SCHED_MAX_COPIES=1, AVX2/AVX512/VBMI/VNNI/AVX_VNNI/BMI2=ON, GGML_CUDA=ON, GRAPHS=ON (compile-time), FA=ON + FA_ALL_QUANTS=ON |
| НЕ собирается | Vulkan, ROCm/HIP/Metal |
| Модель | Qwen3.8-Flash-Next GGUF unsloth UD-Q3_K_XL (3 части), GGUF-архитектура `qwen4exp` (QSA/Lightning Indexer через ggml_top_k, guarded ne0<=1024; gated delta-net + short conv через ggml_ssm_conv; MMoE c experts + shared expert через ggml_mul_mat_id, merged gate_up_exps; per_layer_token_embd PLE; HC_*/PLE_* convs) |
| Запуск | ctx 90000, ctk=f16 ctv=f16, fa=auto, kvu=true, parallel=1, b/ub=1024, t=13 tb=16, cache-ram 3000, ctx-checkpoints 3, lazy-mode=on, load-mode=mlock, fit=off, jinja=on, ot="(.ffn_.*exp|per_layer_token_embd)=CPU", lv=4 |
| CUDA graphs | ВЫКЛЮЧЕНЫ на runtime (compile-time GRAPHS=ON не даёт эффекта) |
| Квант | Q3_K_XL -> стандартный путь dequant Q3_K; KV f16 (без low-bit KV, без KVarN, без TurboQuant) |
| НЕ используется | спекулятивное декодирование (DFlash/MTP/self-spec), vision/mmproj, LoRA, embedding/pooling, TTS, RPC, Docker, Vulkan, ROCm/AMD, Metal, grammar-heavy |
| Дерево пользователя | `D:/dev/llama_compiling/llama.cpp`, HEAD `413cfde3b`, merge-base с upstream `8887a48f0`, ahead 24 |

Цель: ускорение BOTH PP (prefill) и TG (decode).

## Структура каталога

| Файл | Содержание |
|---|---|
| [README.md](README.md) | Это: сетап, легенда, счётчики, ограничения |
| [01-registry.md](01-registry.md) | Индивидуально разобранные коммиты форка, статус каждого |
| [02-applicable.md](02-applicable.md) | Шортлист применимого: что править, где, как |
| [03-already-upstream.md](03-already-upstream.md) | Эквивалент уже есть в upstream/нашем дереве |
| [04-not-applicable.md](04-not-applicable.md) | N/A, сгруппировано по причине |
| [06-build-flags-audit.md](06-build-flags-audit.md) | Аудит флагов сборки: что форк меняет/не меняет |
| [07-raw-inventory.md](07-raw-inventory.md) | Сырой инвентарь (списки коммитов, площади) - НЕ перезаписывать |

## Легенда статусов

| Статус | Значение |
|---|---|
| CUDA-APPLICABLE | Код применим в нашем дереве (CUDA/Windows), нужно портировать |
| CONFIG | Код править не нужно, нужно проверить/поменять конфиг сборки или запуска |
| PORTABLE | Механизм переносим в наше дерево, но требуется ручная адаптация (наше дерево разошлось) |
| UPSTREAM | Эквивалент уже в upstream (и в нашей базе 8887a48f0), переносить нечего |
| BLOCKED | Применимо только после портирования другой подсистемы форка |
| N/A-VULKAN | Vulkan-пути форка |
| N/A-ROCm | ROCm/HIP-специфика |
| N/A-METAL | macOS/Metal |
| N/A-TURBOQUANT | TurboQuant/TCQ/Q2_0-ternary пути (у нас Q3_K_XL, без TurboQuant) |
| N/A-KVARN-LOWBIT | KVarN низкобитный KV cache (у нас ctk=f16 ctv=f16) |
| N/A-SPEC-DECODING | DFlash/MTP/self-speculative (мы не используем спекуляцию) |
| N/A-ARCH | Чужие архитектуры/ARM64/модели |
| N/A-LINUX-ONLY | madvise/POSIX, Linux CI |
| N/A-UNUSED | Корректность фич, которые мы не используем; без эффекта на наш путь |
| N/A-FORK-INTERNAL | Внутренние дела форка: preserve-коммиты дивергенции, фиксы собственных подсистем, CI, build-матрицы |

## Счётчики (индивидуально разобранные коммиты)

| Статус | Кол-во |
|---|---|
| CUDA-APPLICABLE | 2 |
| CONFIG | 1 |
| PORTABLE | 1 |
| UPSTREAM | 4 |
| N/A-TURBOQUANT | 5 |
| N/A-KVARN-LOWBIT | 3 |
| N/A-SPEC-DECODING | 6 |
| N/A-ROCm | 1 |
| N/A-ARCH | 1 |
| N/A-UNUSED | 3 |
| N/A-FORK-INTERNAL | 10 |
| **Итого индивидуально** | **37** |

Плюс укрупнённые группы (не перечислены индивидуальными строками, покрыты
Layer B логами `D:\dev\llama_compiling\bee_audit\log_*.txt`): масса коммитов
KVarN/TurboQuant/TCQ в ggml-cuda (~90+ из 115 логового среза CUDA), семейство
DFlash/SD-* в src/tools (~100+), ROCm/HIP CI и релизы, Vulkan-работы - все они
попадают в N/A-группы по тем же причинам, что и именованные строки реестра.

## Ключевые ограничения реестра

1. Форк diverged mega-коммитом `1a285c4d1` (163 файла, +18279 строк): он
   "сохраняет" TurboQuant/TCQ/KVarN-пути, перетасовывая целые файлы бэкенда.
   Отдельных generic-оптимизаций внутри этих переносов нет; сравнивать diff'ы
   построчно бессмысленно, статус N/A.
2. KV у нас f16 (ctk=f16 ctv=f16, kvu=true): весь KVarN и low-bit KV путь
   (`098e00d34`, `2446c0ed4`, `83429d46a` и масса прочих) неприменим.
3. Q3_K_XL: стандартный Q3_K dequant/GEMM путь. Специальных ускорений Q3_K /
   Q4_K / Q5_K / Q6_K / Q8_0 в CUDA-части форка не найдено (проверено
   `bee_audit\log_cuda_general.txt` + `cuda_files.txt`) - спецпроверка f.
4. Спекулятивное декодирование не используется: всё DFlash/MTP/self-spec
   (включая `f5a7ec15d` mrope, `11f45ed34` graph nodes, `446320565` GPU argmax
   drafter, `f38d36dbf` async GDN tape) - мимо.
5. CUDA graphs выключены на runtime: ускорения путей графа (например `1eaea4494`)
   эффекта не дадут даже если бы не были upstream.
6. Числа эффекта: только цитаты из сообщений коммитов (замеры авторов, обычно
   на ROCm/RTX3090 и не для нашего сетапа); иначе "(оценка)".
7. Наше серверное дерево НЕ содержит подсистем prompt-cache transactional /
   checkpoint budget, которые форк сам же добавлял и потом чинил
   (`16c76b14a`, `fc9ab2bd6`, `c8da45a37`, `13ad92aaa`) - проверка grep по
   [tools/server/server-context.cpp](tools/server/server-context.cpp:1365):
   связи cache_ram <-> checkpoint budget в нашем дереве нет.
