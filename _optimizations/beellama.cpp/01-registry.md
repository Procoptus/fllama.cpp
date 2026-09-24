# Реестр: индивидуально разобранные коммиты Anbeeld/beellama.cpp (вне upstream)

Источник: клон `D:\dev\llama_compiling\_forks_opt\beellama` (HEAD `cd3c41e73`,
merge-base `95ef7fc16`), логи Layer B `D:\dev\llama_compiling\bee_audit\log_*.txt`
(ggml-cuda general, ggml-cpu, ggml core, src, common, tools/server),
`grep_offload.txt`, `grep_topk.txt`. ID = `BE-<sha7>`, ссылка на коммит по ID.
Массовые группы (KVarN/TurboQuant/DFlash/ROCm-CI) не перечислены строками -
они покрыты Layer B логами и сгруппированы в [04-not-applicable.md](04-not-applicable.md).

Итог: CUDA-APPLICABLE 2 | CONFIG 1 | PORTABLE 1 | UPSTREAM 4 | N/A 29
(TURBOQUANT 5, KVARN 3, SPEC 6, ROCm 1, ARCH 1, UNUSED 3, FORK-INTERNAL 10).

## 1. CUDA-APPLICABLE - применимо нам, нужно портировать (2)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [BE-e275191e8](https://github.com/Anbeeld/beellama.cpp/commit/e275191e8) | 2026 | perf: use exp2 SFU fast path for GDN gate decay (спецпроверка a) | Generic CUDA-оптимизация `ggml/src/ggml-cuda/gated_delta_net.cu`, НЕ привязана к KVarN/TurboQuant. Макрос `GDN_EXPF(x) exp2f((x) * 1.442695041f)` вместо `expf` в 6 местах flat+tree GDN-ядер. Наше дерево всё ещё на `expf` ([gated_delta_net.cu:85](ggml/src/ggml-cuda/gated_delta_net.cu:85), [:118](ggml/src/ggml-cuda/gated_delta_net.cu:118), [:132](ggml/src/ggml-cuda/gated_delta_net.cu:132)). GDN используется нашей моделью (delta-net, [delta-net-base.cpp](src/models/delta-net-base.cpp)). Приоритет 1 |
| [BE-86ec665fa](https://github.com/Anbeeld/beellama.cpp/commit/86ec665fa) | 2026 | MMVF/MMF stride-validation guard (спецпроверка h) | Честный вердикт: **только корректность**, НЕ speedup. Валидация `src1->nb` на кратность `2*sizeof(float)` при выборе mmvf/mmf, fallback на совместимое ядро при невыровненном RHS; плюс тест k_v_rhs в test-backend-ops. Может лишь чуть консервативнее выбирать ядро. Портировать для защиты от невалидных float2-чтений. Приоритет 2 |

## 2. CONFIG - правок кода не нужно, проверить конфиг (1)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [BE-f7ca480ed](https://github.com/Anbeeld/beellama.cpp/commit/f7ca480ed) | 2026-07-28 | ci: fix Windows OpenMP loss in release builds (он же db8119cfb) | Не наш CI, но знание бьёт точно в наш сетап: при молча ненайденном OpenMP `ggml_graph_compute` спавнит/джоинит потоки на каждый CPU graph split, фиксовано ~0.5 мс на сплит; в сообщении коммита: "-35..40% tg with --n-cpu-moe" (замер автора, issue #111 форка). У нас CPU-оффлоад экспертов = partial offload с CPU сплитами. Проверить `GGML_USE_OPENMP`/`GGML_OPENMP_ENABLED` в нашем build cache. Детали в [06-build-flags-audit.md](06-build-flags-audit.md) |

## 3. PORTABLE - механизм переносим, нужна ручная адаптация (1)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [BE-297a1cd83](https://github.com/Anbeeld/beellama.cpp/commit/297a1cd83) | 2026-04-15 | optimize copy_cell: batch async D2D copies with single sync point | `src/llama-memory-recurrent.cpp`: 96 отдельных `ggml_backend_tensor_copy` (каждая со своим `cudaStreamSynchronize`) -> очередь `ggml_backend_tensor_copy_async` + одна `ggml_backend_synchronize`. В нашем дереве символа `copy_cell` нет (проверено grep по src/*.cpp - 0 совпадений): функция специфична для fork-пути копирования рекуррентных состояний. Механизм (batch async + single sync) применим к нашему пути копирования GDN-состояний при ресторе/checkpoint. Требует ручного разбора. Потенциал |

## 4. UPSTREAM - эквивалент уже в upstream и в нашей базе (4)

Проверено `git merge-base --is-ancestor <sha> 8887a48f0` (наша merge-base) - все четыре ancestors, файл `bee_audit\anc.txt`.

| ID | PR | Предмет | Эквивалент в upstream |
|---|---|---|---|
| [BE-1eaea4494](https://github.com/Anbeeld/beellama.cpp/commit/1eaea4494) | #21472 | make cuda graphs props check faster (-128 строк, common.cuh + ggml-cuda.cu) | `c5ce4bc22` (ancestor нашей базы). И moot: CUDA graphs у нас выключены на runtime. Детали в [03-already-upstream.md](03-already-upstream.md) |
| [BE-336e225f0](https://github.com/Anbeeld/beellama.cpp/commit/336e225f0) | #21676 | ggml : check return value of CUB calls in argsort and top-k | `009a11332` (ancestor). Только корректность |
| [BE-66c4f9ded](https://github.com/Anbeeld/beellama.cpp/commit/66c4f9ded) | #21168 | ggml-cuda: ds_read_b128 for q4_0 and q4_1 mmq kernels | `66c4f9ded` ancestor нашей базы. Доказательство спецпроверки f: generic k-quant mmq-ускорение уже в базе (q4_0/q4_1; Q3_K своего в форке нет) |
| [BE-e34f04215](https://github.com/Anbeeld/beellama.cpp/commit/e34f04215) | #21665 | CUDA: fuse muls | `e34f04215` ancestor нашей базы |

## 5. N/A-TURBOQUANT (5)

У нас Q3_K_XL + KV f16, TurboQuant/TCQ не используются. Детали и механизмы в
[04-not-applicable.md](04-not-applicable.md).

| ID | Предмет |
|---|---|
| [BE-1010625c5](https://github.com/Anbeeld/beellama.cpp/commit/1010625c5) | turbo4 prefill MMA (спецпроверка e) - fattn.cu только turbo-пути |
| [BE-375555536](https://github.com/Anbeeld/beellama.cpp/commit/375555536) | turbo prefill MMA (продолжение e) |
| [BE-f68ad76b6](https://github.com/Anbeeld/beellama.cpp/commit/f68ad76b6) | FWHT для turbo |
| [BE-1f06a2dfb](https://github.com/Anbeeld/beellama.cpp/commit/1f06a2dfb) | Q2_0 ternary support (новый тип, не Q3_K-хотпуть) |
| [BE-8e6ced0c3](https://github.com/Anbeeld/beellama.cpp/commit/8e6ced0c3) | turbo KV + partial offload |

## 6. N/A-KVARN-LOWBIT (3)

У нас ctk=f16 ctv=f16: весь низкобитный KV путь мимо.

| ID | Предмет |
|---|---|
| [BE-098e00d34](https://github.com/Anbeeld/beellama.cpp/commit/098e00d34) | Generalize KVarN native decode and parallelize CUDA builds |
| [BE-2446c0ed4](https://github.com/Anbeeld/beellama.cpp/commit/2446c0ed4) | q6_0 KV cache (low-bit KV) |
| [BE-83429d46a](https://github.com/Anbeeld/beellama.cpp/commit/83429d46a) | KV-tail work (KVarN-семейство) |

## 7. N/A-SPEC-DECODING (6)

Спекуляцию (DFlash/MTP/self-spec) не используем.

| ID | Предмет |
|---|---|
| [BE-11f45ed34](https://github.com/Anbeeld/beellama.cpp/commit/11f45ed34) | Fix graph number calculation - вся дельта в ветке `LLM_ARCH_DFLASH` ([show_11f45ed34.txt](file:///D:/dev/llama_compiling/bee_audit/show_11f45ed34.txt)), generic-ветку `graph_max_nodes` не трогает |
| [BE-f5a7ec15d](https://github.com/Anbeeld/beellama.cpp/commit/f5a7ec15d) | mrope bug fix - speculative + conversion + dflash пути |
| [BE-446320565](https://github.com/Anbeeld/beellama.cpp/commit/446320565) | GPU argmax for DFlash drafter (ликвидация 15.9MB H2D) - drafter-специфика |
| [BE-f38d36dbf](https://github.com/Anbeeld/beellama.cpp/commit/f38d36dbf) | async tape GDN overlap - spec-путь |
| [BE-91357ddcc](https://github.com/Anbeeld/beellama.cpp/commit/91357ddcc) | argmax topk kernel limit K 32->64 (спецпроверка b) - сделана под DFlash reduced verifier (Gemma4 top_k=64); наш сэмплер top-k=20, лимита не касается |
| [BE-07ac3cec6](https://github.com/Anbeeld/beellama.cpp/commit/07ac3cec6) | merge PR#25 fix-argmax-topk-64 (то же, merge-коммит) |

## 8. N/A-ROCm (1)

| ID | Предмет |
|---|---|
| [BE-8b250fba2](https://github.com/Anbeeld/beellama.cpp/commit/8b250fba2) | Fix HIP FA launch bounds and Windows CPU packaging - HIP-часть нас не касается |

## 9. N/A-ARCH (1)

| ID | Предмет |
|---|---|
| [BE-0b035b3a2](https://github.com/Anbeeld/beellama.cpp/commit/0b035b3a2) | ARM64 fix |

## 10. N/A-UNUSED (3)

Корректность/fallback для путей, которых у нас нет в_graph'е.

| ID | Предмет |
|---|---|
| [BE-0ef8c42f4](https://github.com/Anbeeld/beellama.cpp/commit/0ef8c42f4) | reject unsupported BF16 scale - корректность BF16-пути (у нас f16/f32) |
| [BE-9f268fb03](https://github.com/Anbeeld/beellama.cpp/commit/9f268fb03) | reject unsupported CPU FA types - FA у нас на GPU |
| [BE-76e3f43de](https://github.com/Anbeeld/beellama.cpp/commit/76e3f43de) | cpu: support f16 out_prod fallback - feature-add fallback CPU out_prod f16 |

## 11. N/A-FORK-INTERNAL (10)

| ID | Предмет |
|---|---|
| [BE-1a285c4d1](https://github.com/Anbeeld/beellama.cpp/commit/1a285c4d1) | preserve TurboQuant and TCQ backends - mega-коммит дивергенции, 163 файла +18279/-681; включает gated_delta_net.cu +558, ssm-conv.cu +128, argmax.cu 486, topk-moe.cu, set-rows.cu - preserve/перенос чужих путей, не самостоятельные generic-оптимизации |
| [BE-16c76b14a](https://github.com/Anbeeld/beellama.cpp/commit/16c76b14a) | prompt-cache transactional subsystem, 35 файлов +3513 - подсистема форка (KVarN-центричная), у нас её нет |
| [BE-fc9ab2bd6](https://github.com/Anbeeld/beellama.cpp/commit/fc9ab2bd6) | продолжение той же подсистемы, 19 файлов +785 |
| [BE-c8da45a37](https://github.com/Anbeeld/beellama.cpp/commit/c8da45a37) | Fix prompt-cache host budget enforcement - фикс собственной подсистемы |
| [BE-13ad92aaa](https://github.com/Anbeeld/beellama.cpp/commit/13ad92aaa) | stop deriving checkpoint budget from cache-ram - фикс собственной подсистемы; в нашем дереве связи cache_ram <-> checkpoint budget нет (grep по [server-context.cpp](tools/server/server-context.cpp:1365) - только `n_ctx_checkpoints`, бюджет-выводов из cache-ram нет) |
| [BE-ac792986d](https://github.com/Anbeeld/beellama.cpp/commit/ac792986d) | preserve fused GDN 4D state fast path - compat-прослойка старого 4D-формата state (S_v,S_v,H,n_seqs) рядом с новым 3D snapshot API. Дельта: ggml.h/+1 комментарий, ggml.c +13/-5 assert'ы обеих раскладок, ops.cpp +9/-3 детект 4D, delta-net-base.cpp -3 (убран reshape). Наше дерево уже на 3D snapshot API ([ggml.c:6316+](ggml/src/ggml.c:6316)); 4D-compat нам не нужен (спецпроверка g) |
| [BE-5eaba174c](https://github.com/Anbeeld/beellama.cpp/commit/5eaba174c) | reduce default FA pair matrix - build-матрица FA под типы форка |
| [BE-df8933c26](https://github.com/Anbeeld/beellama.cpp/commit/df8933c26) | Allow ignoring uncompiled CUDA FA pairs - plumbing сборки матрицы форка |
| [BE-a596a5c72](https://github.com/Anbeeld/beellama.cpp/commit/a596a5c72) | HALF_QUANTS build mode - флаг сборки форка (в [06-build-flags-audit.md](06-build-flags-audit.md)) |
| [BE-94605e9fe](https://github.com/Anbeeld/beellama.cpp/commit/94605e9fe) | ci: seed version caches from predecessor branches - релизный CI форка |

## Специальные проверки (a)-(h): ответы

| Проверка | Ответ |
|---|---|
| (a) e275191e8 GDN exp2 | **Generic CUDA, применим.** Патчит только `ggml/src/ggml-cuda/gated_delta_net.cu` (flat + tree ядра, 6 мест), без KVarN/TurboQuant-зависимостей. Наше дерево на expf (3 места найдено grep: строки 85, 118, 132 - наше дерево не содержит fork-only tree-ядра). VALUABLE для TG/PP delta-net |
| (b) CUDA top-k/radix для QSA indexer | **Нет generic-оптимизаций.** Всё найденное по top-k/argsort/radix: фикс лимита K 32->64 для sampling topk_f32 под DFlash-verifier (BE-91357ddcc), upstream CUB return-value чек (#21676), DFlash selector plumbing. Алгоритмических ускорений `ggml_top_k` (Lightning Indexer) в форке нет |
| (c) MoE CPU-offload / overlap / pinned | **Нет generic-работы.** Ни cudaMallocHost/pinned, ни async overlap для mul_mat_id-оффлоада. madvise - POSIX (N/A-LINUX-ONLY по определению). Единственное близкое: BE-297a1cd83 async-batch копирований (PORTABLE) |
| (d) 1eaea4494 cuda graphs props | **UPSTREAM** = c5ce4bc22 (#21472), ancestor нашей базы. Moot: graphs выключены на runtime |
| (e) 1010625c5 / 375555536 | **N/A-TURBOQUANT.** Статы diff'ов: только turbo-секции fattn.cu. Механизм: bulk-dequant turbo-квантованных K/V в fp16 перед MMA в prefill. Цитаты из сообщений: "pp4096 588->1113 tok/s (1.9x)", "1.78x (98.8% of q8_0)", стенд RTX 3090 Qwen3.5-27B Q6_K |
| (f) standard k-quant CUDA (Q3_K и др.) | **Ниха нет.** В 115-коммитном CUDA-срезе generic k-quant speedup'ов нет; только новые типы (Q2_0 ternary, q6_0 KV). Совпадение: upstream #21168 ds_read_b128 q4_0/q4_1 mmq уже в нашей базе |
| (g) delta-net/ssm/conv1d | **Один чистый: e275191e8 (см. a).** ac792986d - API-compat форка, нам не нужен (дерево на 3D API). ssm-conv.cu менялся только внутри mega-preserve 1a285c4d1. CPU-оптимизаций conv1d/GDN в ggml-cpu не найдено |
| (h) 86ec665fa MMVF/MMF guard | **Только корректность.** Валидация шагов src1 (кратность 8 байт) + vals_per_T проверка; fallback при невыровненном RHS. Это не ускорение; может сделать выбор ядра чуть более консервативным. Портить стоит ради защиты от невалидных float2-чтений |

## Потенциальные, требует ручного разбора (без цифр)

- BE-1a285c4d1 внутри содержит перенос tree-вариантов GDN-ядер (gated_delta_net.cu
  +558 строк) и ssm-conv.cu +128: алгоритмически это параллельные/tree GDN-ядра,
  которых нет в нашей базе; но они "заспайшены" в mega-preserve с turbo-путями.
  Ручной разбор: выделить tree-GDN и проверить совместимость с нашим 3D-state API.
  Оценка эффекта: "(оценка, не измерено)".
- BE-297a1cd83 (PORTABLE, см. выше).
- BE-5678fc3b8 "restore D512 FA selector" - восстановление selector'а после
  апстрима; к нашему D512/QSA-пути отношения не имеет (в реестр не включена как
  отдельная строка: fork-override removal).

## Итог

Для CUDA + qwen4exp + f16 KV + Q3_K_XL изолированных применимых элементов:
**CUDA-APPLICABLE 2 + CONFIG 1 + PORTABLE 1 = 4** (из них реальных speedup-кандидатов:
e275191e8 (GDN exp2) и проверка OpenMP). beellama в основном про
ROCm/KVarN/TurboQuant/DFlash - это честный результат аудита.
