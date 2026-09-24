# Сводный отчёт: оптимизации из форков для qwen4exp на CUDA sm_86

Дата: 2026-09-04.

Два форка, полный аудит каждого в дочерних каталогах:
- [CachyLLama](CachyLLama/README.md) - HEAD `8dca17b4dc64756dd178f211e224ebe19d5c21f9` (2026-09-01), merge-base `3466812d1`, ahead 165 (из них 144 non-merge, все разобраны индивидуально), behind 40.
- [beellama.cpp](beellama.cpp/README.md) - HEAD `cd3c41e73d0a5deb1d2051bec04df73673a40e9c` (2026-09-01), merge-base `6fdd0ac8907fd973a42b876357823ad2124cd8ed` (2026-08-27), ahead 915, behind 132, non-merge 848; ref сравнения upstream/master `95ef7fc16054e63b427a3ef00188e055ef7586d8` (2026-09-03). Индивидуально разобрано 37, остальное покрыто группами (см. beellama/05).
- Числа форков приведены "как есть" из соответствующих 07-raw-inventory.md.

Сетап (оба аудита): Windows 11 25H2; i7-12700K (E-cores отключены, AVX-512 разблокирован, нет AVX512-BF16); RTX 3080 10GB (sm_86); 64GB DDR4-3600. Сборка CMake+Ninja Release MSVC, флаги: `-DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DGGML_NATIVE=ON -DGGML_LTO=ON -DGGML_CCACHE=ON -DGGML_OPENMP=ON -DGGML_SCHED_MAX_COPIES=1 -DGGML_AVX2=ON -DGGML_AVX512=ON -DGGML_AVX512_VBMI=ON -DGGML_AVX512_VNNI=ON -DGGML_AVX_VNNI=ON -DGGML_BMI2=ON -DGGML_CUDA=ON -DGGML_CUDA_GRAPHS=ON -DGGML_CUDA_FA=ON -DGGML_CUDA_FA_ALL_QUANTS=ON`. Цель - llama-server.

Модель: Qwen3.8-Flash-Next unsloth UD-Q3_K_XL (3 части), GGUF-архитектура qwen4exp (QSA/Lightning Indexer -> ggml_top_k; delta-net + short conv -> ggml_ssm_conv; MMoE + shared expert -> ggml_mul_mat_id; PLE per_layer_token_embd). Запуск: ctx 90000, `-ctk f16 -ctv f16`, fa=auto, kvu=true, parallel=1, b/ub 1024, t 13, tb 16, cache-ram 3000, ctx-checkpoints 3, lazy-mode on, load-mode mlock, ot="(.ffn_.*exp|per_layer_token_embd)=CPU". CUDA graphs на рантайме ВЫКЛЮЧЕНЫ (compile-time ON недействующий). Не используются: spec decoding (DFlash/MTP), Vulkan, ROCm, Metal, Vision, LoRA, RPC, TTS, embedding.

Наше дерево: HEAD `413cfde3b`, merge-base upstream `8887a48f0`, ahead 24 (собственные порты: fast tensor names, fused rms_norm CUDA, FA mma f16 ncols2 head256/gqa12, small F32 GEMM fast path, QSA top_k-skip, IQP AVX512/VNNI Q5_K/Q6_K, AVX-VNNI MSVC detect, NUMA fallback, PR-25346, PR-27590, pr-27402-iqp).

Итог поиска: индивидуально пригодных кандидатов **13** (CachyLLama 9 CUDA-APPLICABLE + beellama 4: 2 CUDA-APPLICABLE, 1 CONFIG, 1 PORTABLE). Сводный шортлист - **11** строк главной таблицы (таблица ниже) + 1 отклонённая строка; из 11 дешёвых (сложность "низкая" или без кода) - **8**. Остальные коммиты обоих форков: BLOCKED-подсистемы, чужие бэкенды, не используемые фичи, внутреннее дела форков (см. 04/05 каждого каталога).

## 1. Главная таблица оптимизаций

| № | Источник | PR/Коммит | Что оптимизирует | Где править | Ожидаемый эффект | Сложность | Риск | Статус |
|---|---|---|---|---|---|---|---|---|
| 1 | CachyLLama | CL-a657590ae | true sparse gather для QSA attention: dense-маску по всем n_kv заменить выборкой n_sel строк через ggml_get_rows + flash_attn_ext по [d,1,n_sel,nt] | src/models/qwen4exp.cpp (build_attn_qsa, строка 695) + новые qsa_gather_n_sel()/build_qsa_top_k()/build_attn_qsa_gather() + src/models/models.h | ~40x меньше work attention на 131K при n_kv~100K, indexer_top_k=2048, compress_ratio=4 -> n_sel=2304 (цифра из сообщения автора commit, не измерено у нас) | высокая | средний (env LLAMA_QSA_GATHER, дефолт 16384; только decode, FA on, без alibi) | портровать, главный кандидат |
| 2 | beellama.cpp | BE-e275191e8 | exp2 SFU fast path для GDN gate decay: `#define GDN_EXPF(x) exp2f((x) * 1.442695041f)` вместо expf | ggml/src/ggml-cuda/gated_delta_net.cu (у нас expf(*g_t) строка 85, expf(g_t[i]) строки 118,132; из 6 мест форка наши 3) | доли процента на слой (оценка, не измерено) | низкая | низкий | портровать; проверить test-backend-ops -o GATED_DELTA_NET |
| 3 | CachyLLama | CL-f658fc5af | guard `n_parallel <= 1` вокруг prompt save/load/update в get_available_slot (у нас не защищено, проверено) | tools/server/server-context.cpp строки 1625-1646 | ~1 с/ход и ~22 GiB VRAM<->RAM на 40 мин при 128K f16 (автор); config-эквивалент `--cache-ram 0` | низкая | низкий | портровать (2 строки guard) |
| 4 | CachyLLama | CL-b83d23022 | seq_rm recurrent memory: частичный rollback `return false` -> fall-through в cell-cleanup (краш streaming+tools) | src/llama-memory-recurrent.cpp функция seq_rm (строка 161), целевой return false строка 203 | фикс краша, не скорость | низкая (1-2 строки) | низкий | портровать первым; тест test-recurrent-state-rollback уже в дереве |
| 5 | CachyLLama | CL-67f82f589 | backend splits резать по константе GGML_SCHED_MAX_SPLIT_INPUTS, а по выросшей inputs_capacity | ggml/src/ggml-backend.cpp строка 1343 (комментарий-мишень на 1341; оригинал nathanw1014 76ad2ba9f) | минус пик compute buffer, плюс TG (оценка, не измерено) | низкая | низкий (массив оставить growable) | портровать |
| 6 | CachyLLama | CL-024dff234 | evict чекпойнт с наибольшим pos_min, а не front() (insertion-oldest) | tools/server/server-context.cpp строки 2317-2322; LCP-порог pos_min_thold строка 3270; список std::list<common_prompt_checkpoint> tools/server/server-task.h строка 569 | меньше полных репроцессов промптов; цифра 28123 токенов - (цифра не подтверждена), см. 6(д) | средняя (портить только выбор жертвы, 10-20 строк; fork-версия зависит от отсутствующего deferred_create_final_checkpoint) | средний | портровать частично |
| 7 | CachyLLama | CL-022c16ed5 | защитный guard на K-shift NULL host buffer (у нас GGML_ASSERT без null-guard) | src/llama-kv-cache.cpp строка 1527 GGML_ASSERT (set_input_k_shift, начало 1526); conv_hash-часть BLOCKED | защита от краша; сценарий на qwen4exp невозможен (RoPE-слои есть) | низкая | низкий | опционально, низкий приоритет (оценка, не измерено) |
| 8 | CachyLLama | CL-dd3fccf1a | CONFIG: DSA indexer key cache держать f16 (у автора деградация dispatch 0.8ms->97ms при 12k, Vulkan) | конфиг запуска | правило уже выполнено: `-ctk f16 -ctv f16` | нет кода | нет | проверить и не включать квантованный KV без перепроверки indexer fast-path |
| 9 | beellama.cpp | BE-f7ca480ed | CONFIG: OpenMP на MSVC может молча теряться; без GGML_USE_OPENMP ggml_graph_compute поднимает одноразовый threadpool на каждый CPU graph split ~0.5 мс фиксированно | сборка (CMake cache) | автор измерял "-35..40% tg с --n-cpu-moe" на своём сетапе; у нас кэш проверен: GGML_OPENMP:BOOL=ON (см. раздел 7) | нет кода | нет | чек-лист: пересобирать из Developer PowerShell |
| 10 | beellama.cpp | BE-86ec665fa | correctness-guard MMVF/MMF: проверка `src1->nb` кратно `2*sizeof(float)`, helper ggml_cuda_mmvf_rhs_compatible() + тест k_v_rhs | ggml/src/ggml-cuda/mmvf.cu, ggml/src/ggml-cuda/mmf.cu | не скорость; защита наших CPU-offload src1- layouts (ot=...=CPU) | низкая (порт as-is) | низкий | портровать; test-backend-ops -o MUL_MAT_ID / -o MUL_MAT |
| 11 | beellama.cpp | BE-297a1cd83 | батч async D2D копий: 96 ggml_backend_tensor_copy с cudaStreamSynchronize каждое -> async queue + одна sync | аналог copy_cell у нас НЕ существует (grep `copy_cell` по дереву пуст); прототип - файл форка src/llama-memory-recurrent.cpp (+36/-3 по git show) | (оценка, не измерено) | средняя (ручная адаптация) | средний | портовать вручную по прототипу |
| 12 | beellama.cpp | BE-91357ddcc | topk K 32->64 | - | нет: наш top_k = 20; задача была у верификатора DFlash/Gemma4 | - | - | ОТКЛОНЕНО (см. beellama/05) |

## 2. Заблокированные и крупные подсистемы (не портировать)

Подробные вердикты: [CachyLLama/05](CachyLLama/05-blocked-or-needs-diff.md), [beellama.cpp/05](beellama.cpp/05-blocked-or-needs-diff.md). Кратко:

- CL-6a0db500c SSD-backed KV cache (kv_page_manager 'KVPG'): diff ~+5335/-1357 (~6400 строк), POSIX-блокировки, за ней BLOCKED-цепочка ~48 коммитов Cachy. Не портировать.
- CL-56dca0825 + CL-cc60f8912 global system prompt KV cache: держится на SSD-подсистеме.
- API seq_rm_attn_only (CL-54e51d11f, CL-5dd48c54b, CL-757361a70): блокированы тем же; первое, что взять, если SSD когда-нибудь портируем.
- TurboQuant/TCQ prefill MMA (1010625c5 "pp4096 588->1113 tok/s (1.9x)", 375555536 "1.78x, 98.8% of q8_0"): цифры автора на TurboQuant-кэше; у нас KV f16, типа нет. N/A.
- KVarN (fattn-kvarn-dispatch.cu): отдельный FA-бэкенд форка, у нас нет диспетчера. N/A.
- DFlash/SD-* семейство (~100+ коммитов beellama: `dflash` 72 в subject, f1a9b5a19, 5ecbe1ac1 и др.): spec decoding у нас выключен. N/A.
- preserve-мегамассы (1a285c4d1 - 163 файла; 80bb3c794 - 56; 686e63aa3 - 118): merge-preserving, оптимизаций не содержат.
- c8da45a37 / 13ad92aaa (prompt-cache бюджеты на сервере форка): семантика отделилась от upstream, только ручной разбор отдельной задачей; эквивалент закрывается строками 3 и 6 таблицы.
- ac792986d GDN 4D state: наша база на 3D snapshot API, NO-DIFF.

## 3. Выводы по флагам сборки

По обоим 06-build-flags-audit.md:

- Ни один флаговый change обоих форков нам не нужен: их cmake-правки привязаны к их подсистемам (KVarN, TurboQuant, DFlash, Vulkan/ROCm тюнинг) или к Linux CI.
- Наш профиль согласован: Release, arch 86, NATIVE ON, AVX-512/VBMI/VNNI ON (BF16 нет и не просим - i7-12700K его не имеет), CUDA_GRAPHS ON на компиляции (на рантайме выключены, флажок безвреден), FA_ALL_QUANTS ON (KV f16, тоже безвреден).
- Единственный действующий вывод - строка 9 таблицы: OpenMP на MSVC может молча потеряться; при пересборке использовать Developer PowerShell и проверять cache.

## 4. Конфиг- only пункты (без правок кода)

- Строка 8: не включать квантованный KV (`-ctk/-ctv` не f16) без перепроверки fast-path lightning indexer.
- Строка 9: OpenMP в cache должен быть ON; при propag-потере - пересборка из Developer PowerShell.
- Строка 3 имеет config-эквивалент: `--cache-ram 0` (приемлемо, если чекпойнты промптов не нужны; тогда строка 6 теряет смысл).
- Строка 1 активируется env `LLAMA_QSA_GATHER` (дефолт 16384) после порта кода.

## 5. Уже в upstream / NO-DIFF

- CachyLLama: 1 UPSTREAM (CL-4db9548a8 IMROPE shift/reuse guard - у нас src/llama-kv-cache.cpp:1189-1198 (get_can_shift), класс llama_memory_hybrid_idx, ISWA не instantiated; см. CachyLLama/03).
- beellama.cpp: 4 UPSTREAM (см. beellama/03).
- NO-DIFF позиции: CL-f629077d1 (ISWA reclass), ac792986d, 91357ddcc, несуществующий 6d87d421c - см. оба 05.

## 6. Фактические коррекции (зафиксированы при аудите)

(а) CachyLLama SHA-предупреждение: `4892d5791`, `e2234e824`, `a5889b2d8` НЕ существуют в форке; реальные: PR#1 = 7b9afec66, PR#2 = a2b13f5ab, PR#3 = 8b2cf6c66.
(б) beellama merge-base = `6fdd0ac89` (2026-08-27); `95ef7fc16` (2026-09-03) - это ref сравнения upstream/master, а не merge-base (как в 07-raw-inventory.md).
(в) `6d87d421c` (mrope graph hoisting) не существует ни в клоне beellama, ни в клоне CachyLLama: `git show --stat` -> `fatal: unknown revision` (2026-09-04); в инвентаре 07 его нет. В запросе он фигурировал кандидатом - исключён, (цифра не подтверждена).
(г) beellama.cpp НЕ поддерживает qwen4exp (ni qwen4exp / LLM_ARCH_QWEN4EXP / build_attn_qsa в дереве форка) - его модельные оптимизации к нашей модели применить нельзя напрямую.
(д) Две коррекции по требованию заказчика:
  1. Число "28123" для CL-024dff234 НЕ подтверждено. В CachyLLama/02-applicable.md строка 146-147 цитата автора "Observed on task #270: full reprocess of 28123 tokens." сохранена как цитата, но в выводах помечена "(цифра не подтверждена)".
  2. В нашем дереве НЕТ gather-операции для KV. Слово "gather" в src/models/qwen4exp.cpp встречается только в комментариях (строки 110, 169, 171, 609, 1037, 1144, 1197). После ggml_top_k (src/models/qwen4exp.cpp:684) attention идёт dense-маской по всем n_kv в build_attn_qsa (src/models/qwen4exp.cpp:695). То есть строка 1 таблицы - это НОВАЯ операция, а не ускорение существующей.

## 7. Проверка измерений и артефактов сборки

Замеры скоростей в этом аудите НЕ проводились (запрет на билды/прогоны); все проценты и тайминги - цитаты авторов с пометками "(оценка, не измерено)", если эффект не из замера автора на эквивалентном сетапе.

Проверено наличие артефактов (read-only, 2026-09-04):
- `build_avx_512\bin\` существует; в нём llama-server.exe, llama-bench.exe, test-backend-ops.exe, test-recurrent-state-rollback.exe и ~90 прочих exe.
- `build_avx_512\CMakeCache.txt`: `CMAKE_BUILD_TYPE:STRING=Release` (строка 70), `CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=86` (73), `GGML_CUDA_GRAPHS:BOOL=ON` (680), `GGML_NATIVE:BOOL=ON` (770), `GGML_OPENMP:BOOL=ON` (788).

Команды для повторной проверки из PowerShell:

```powershell
Get-ChildItem build_avx_512\bin | Select-Object Name, Length
Select-String -Path build_avx_512\CMakeCache.txt -Pattern 'GGML_OPENMP|GGML_NATIVE|GGML_CUDA_GRAPHS|CMAKE_CUDA_ARCHITECTURES|CMAKE_BUILD_TYPE'
```

Тесты для верификации портов (после сборки пользователем): `test-backend-ops -o GATED_DELTA_NET` (строка 2), `-o MUL_MAT_ID` и `-o MUL_MAT` (строка 10), `test-recurrent-state-rollback` (строка 4), llama-bench до/после строк 1/5.

## 8. Рекомендуемый порядок внедрения

Дешёвые и стабилизирующие - первыми, большой gather - последним (он же самый выигрышный):

1. Строка 4, CL-b83d23022 - фикс краша, 1-2 строки.
2. Строки 8 и 9 - конфиг-проверки без кода (OpenMP ON в кэше уже подтверждён).
3. Строка 3, CL-f658fc5af - guard из 2 строк (или временно `--cache-ram 0`).
4. Строка 2, BE-e275191e8 - макрос GDN_EXPF, 3 места.
5. Строка 5, CL-67f82f589 - константа вместо ёмкости.
6. Строка 10, BE-86ec665fa - guard as-is + тесты MUL_MAT*.
7. Строка 6, CL-024dff234 - только выбор жертвы (10-20 строк).
8. Строка 7, CL-022c16ed5 - защитный guard, попутно.
9. Строка 1, CL-a657590ae - основной кандидат; порт helper'ов + модели, тест на длинном контексте decode.
10. Строка 11, BE-297a1cd83 - ручная адаптация async-копий по прототипу форка.

Счётчики реестров не изменялись: CachyLLama 144 (CUDA-APPLICABLE 9, UPSTREAM 1, BLOCKED 48, N/A-VULKAN 26, ROCm 4, METAL 2, ARCH 4, LINUX-ONLY 15, UNUSED-FEATURE 18, FORK-INTERNAL 16, ISWA-PATH 1), beellama.cpp 37 (CUDA-APPLICABLE 2, CONFIG 1, PORTABLE 1, UPSTREAM 4, N/A-TURBOQUANT 5, KVARN-LOWBIT 3, SPEC-DECODING 6, ROCm 1, ARCH 1, UNUSED 3, FORK-INTERNAL 10).
