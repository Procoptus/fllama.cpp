# Реестр проверенных оптимизаций из ikawrakow/ik_llama.cpp

Дата проверки: 2026-09-02
Правка 2026-09-02 (вторая): отменена рекомендация `-DGGML_AVX512_BF16=ON` (CPU без AVX512-BF16,
под cl.exe не компилируется). Подробности: `06-build-flags-audit.md`, раздел «BF16: отмена
рекомендации».
Проверяемое дерево: upstream llama.cpp, база `b10740-26-gead916c05`, HEAD `ead916c05`, + 4 локальных коммита
Цель: ускорение PP и TG

Назначение каталога: это накопительный реестр. Перед проверкой нового PR из
ik_llama.cpp сначала искать его ID здесь (`checked-ids.txt`). Если ID уже есть,
повторно не анализировать, смотреть вердикт.

## Целевой сетап

| Параметр | Значение |
|---|---|
| CPU | Intel i7-12700K (Alder Lake-S, Model 151), E-ядра отключены (8P / 16 потоков), AVX-512 разблокирован микрокодом; AVX512-BF16 нет |
| GPU | NVIDIA RTX 3080 10 GB, sm_86, одна карта |
| RAM | 64 GB DDR4-3600 |
| OS | Windows 11 25H2 (26200.9168) |
| Компилятор | MSVC 14.44.35207 + Ninja, CMAKE_BUILD_TYPE=Release |
| Модель | qwen3.8-Flash-Next, квант Q3_K_XL, GGUF из 3 частей |
| Архитектура | `qwen4exp`, head_dim=256, gqa_ratio=12 |
| Конфиг запуска | ctx=90000, ctk=f16, ctv=f16, b=1024, ub=1024, t=13, tb=16, `-ot "(.ffn_.*exp\|per_layer_token_embd)=CPU"`, parallel=1, ctx-checkpoints=3, fa=auto (по логам FA работает), kvu=true, lazy-mode=on, load-mode=mlock, cache-ram=3000, fit=off, jinja=on, temp=1.0, top-p=0.95, top-k=20 |
| Runtime env | `GGML_CUDA_GRAPH_OPT=1` задан в батнике (подтверждено пользователем) |

## Структура каталога

| Файл | Содержимое |
|---|---|
| `01-registry.md` | Главная таблица: все проверенные PR/коммиты и вердикт |
| `02-applicable.md` | Развёрнутый разбор того, что применимо (что править, где, как) |
| `03-already-upstream.md` | Эквивалент уже есть в upstream. Повторно не проверять |
| `04-not-applicable.md` | Не применимо, сгруппировано по причине |
| `05-blocked-or-needs-diff.md` | Заблокировано зависимостью либо дифф не верифицирован |
| `06-build-flags-audit.md` | Аудит CMake-флагов + очищенная команда сборки |
| `07-raw-inventory.md` | Сырые числовые заявления из инвентаря ik |
| `checked-ids.txt` | Плоский список ID для фильтрации: `findstr /i 2225 checked-ids.txt` |

## Легенда статусов

| Статус | Значение |
|---|---|
| `APPLIED` | Уже портировано в локальное дерево вручную |
| `CONFIG` | Код уже в upstream, нужен только флаг или параметр запуска |
| `PORTABLE` | Код нужно портировать вручную |
| `UPSTREAM` | Эквивалент уже есть в upstream, переносить нечего |
| `N/A` | Не применимо к сетапу, архитектуре или платформе |
| `BLOCKED` | Применимо только после другой оптимизации либо нет нужного API |
| `NO-DIFF` | Вердикт по описанию PR, дифф не проверялся |

## Счётчики

| Статус | Количество |
|---|---|
| APPLIED | 4 |
| CONFIG | 5 PR + 3 неб-PR рычага |
| PORTABLE | 5 |
| UPSTREAM | 15 |
| BLOCKED | 1 |
| N/A | 46 |
| **Итого уникальных ID** | **~76** |

Приоритетных позиций (ожидаемый эффект выше погрешности замера): `#2225`, `#1599`,
`#1261`, `#1137`/`#1403` (только переконвертация GGUF), `#1049`.

## Ключевые ограничения дерева (действуют на весь реестр)

1. Ветки `pr-2373` с историей ik имеют пустой merge-base с master. `cherry-pick`
   невозможен, только ручные патчи.
2. В дереве нет каталога `ggml/src/iqk/`. Все IQK-оптимизации ik непереносимы как код.
3. `head_dim=256`, `gqa_ratio=12`. Кандидаты под head128/GQA16/GQA10/head512 отпадают.
   Наш случай уже покрыт локальным коммитом `8377362c7` (`ggml/src/ggml-cuda/fattn.cu:91`).
4. CUDA Graphs НЕ отключаются MoE на sm_86: `ggml_cuda_mul_mat_id_needs_sync()`
   возвращает false, потому что `ggml_cuda_should_use_mmq()` всегда true для Turing+.
5. qwen4exp не имеет MTP/nextn (`llama_model_n_layer_nextn`, `src/llama-model.cpp:2740`).
   Из speculative decoding доступен только ngram self-speculation.
6. QSA индексер вызывает `ggml_top_k` на CPU по строкам до n_kv=90000. GPU путь
   отключён guard'ом `ne0<=1024` (`ggml/src/ggml-cuda/ggml-cuda.cu:5360`), CUB не найден.
7. `--fit` и ручной `-ot` взаимоисключающие (`common/fit.cpp:483` бросает исключение).
8. Дельта-нет полностью реализован на GPU (`ggml/src/ggml-cuda/gated_delta_net.cu`),
   shared memory не использует.
