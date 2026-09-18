# Реестр: все проверенные PR/коммиты ik_llama.cpp

Группа проверки: `G1` = CUDA/Graphs/MoE/KV/FA, `G2` = loader/RAM/CPU/AVX512/delta-net/spec/sampling,
`G3` = финальная группа из 24 кандидатов + хвост инвентаря.

Ссылка на PR: `https://github.com/ikawrakow/ik_llama.cpp/pull/<ID>`

## 1. APPLIED - уже портировано локально

| ID | Тема | Область | Статус | Эффект | Сложн. | Риск | Группа |
|---|---|---|---|---|---|---|---|
| #2297 | правка FA | CUDA / FlashAttention | APPLIED (`c84ce61fc`) | - | - | - | G1 |
| #2372 | FA DKQ=256 + gqa=12 -> ncols2=4 | CUDA / FlashAttention / qwen4exp | APPLIED (`8377362c7`) | +FA для нашей модели | - | - | G1 |
| #2373 | `k_simple_gemm_f32` fast path в `ggml_cuda_mul_mat()` | CUDA / PP | APPLIED (`ef5ac8948`) | +PP | - | - | G1 |
| #860 | `ggml_format_name_fast` вместо `vsnprintf` в горячих путях имён | CPU / build-overhead / TG | APPLIED (`413cfde3b`) | ~1 % TG | низкая | низкий | G3 |

## 2. CONFIG - код уже в upstream, нужен только параметр

| ID | Тема | Область | Статус | Эффект | Сложн. | Риск | Группа |
|---|---|---|---|---|---|---|---|
| #1599 | смешанный / квантованный KV cache (`-ctk`/`-ctv`) | KV-cache / VRAM / FA | CONFIG | -40..50 % VRAM на KV, +2..6 % PP/TG косвенно | нет | средний (качество) | G2 |
| #1261 | self-speculative ngram | TG / speculative decoding | CONFIG | +10..25 % TG на шаблонном тексте, ~0 на креативном | нет | средний | G2 |
| #2262 | `-ncmoe` вместо ручного `-ot` regex | MoE / VRAM / UX | CONFIG | 0 % скорости, снимает ручной подбор, открывает `--fit` | нет | нет | G1 |
| #1137 | merged gate+up experts (`ffn_gate_up_exps`) | MoE / PP / VRAM | CONFIG (только переконвертация GGUF) | +3..8 % PP на CPU-оффлоаде экспертов | нет | нет | G2, G3 |
| #1403 | fused MoE (в части merged gate/up) | MoE / PP | CONFIG (только переконвертация GGUF) | см. #1137 | нет | нет | G2, G3 |
| - | `GGML_CUDA_GRAPH_OPT=1` как env при запуске | CUDA Graphs | CONFIG, **уже сделано пользователем** | +1..4 % TG | нет | нет | G2 |
| - | `-DGGML_AVX512_BF16=ON` в cmake | CPU / AVX-512 / сборка | N/A, ОШИБКА ИСПРАВЛЕНА 2026-09-02 (было CONFIG) | 0 % | нет | см. `06-build-flags-audit.md` L118-151 | G2 |
| - | `-t` / `-tb` тюнинг при отключённых E-ядрах | CPU / threading | CONFIG | +0..5 % PP | нет | низкий | G2 |
| - | `-fa on` vs `auto` (FA по логам уже работает) | CUDA / FA | CONFIG, проверка ради baseline | +0..3 % PP | нет | низкий | G1 |

## 3. PORTABLE - нужно портировать код

| ID | Тема | Область | Статус | Эффект | Сложн. | Риск | Группа |
|---|---|---|---|---|---|---|---|
| #2225 | bucket top_k вместо `std::partial_sort` | CPU / TG / QSA индексер | PORTABLE, **наивысший приоритет** | +2..5 % TG на длинном контексте (-20..40 % времени индексера) | средняя | средний | G2, G3 |
| #1049 | `needs_sync[]` в `ggml_backend_sched_compute_splits` | CUDA / CPU scheduling / TG+PP | PORTABLE | +1..3 % TG/PP | средняя | средне-высокий | G3 |
| #1427 | перепись CPU `rms_norm` (SIMD, row-parallel) | CPU / AVX-512 | PORTABLE, низкий приоритет | ~0 % при текущем оффлоаде, +1..2 % PP при расширении CPU-части | низкая-средняя | низкий | G2, G3 |
| #1417 | `top_n_sigma`: `pow(d,2)` -> `d*d` | Sampling | PORTABLE | 0 % у нас (сэмплер выключен) | низкая | низкий | G2, G3 |
| #1560 | CLI-флаг для резерва compute buffer по худшему графу | VRAM / стабильность | PORTABLE (только CLI, механизм уже в upstream) | 0 % скорости | низкая | низкий | G3, NO-DIFF |

## 4. UPSTREAM - эквивалент уже есть, переносить нечего

| ID | Тема | Область | Статус | Где в upstream | Группа |
|---|---|---|---|---|---|
| #2316 | graph UID + сравнение адресов src | CUDA Graphs | UPSTREAM | `ggml/src/ggml.c:56`, `ggml/src/ggml-impl.h:345`, `ggml/src/ggml-backend.cpp:1085` | G1 |
| #2136 | повторный capture при CPY | CUDA Graphs | UPSTREAM | `ggml/src/ggml-cuda/ggml-cuda.cu:2624` | G1 |
| #2292 | трёхуровневая защита padding | CUDA / quant kernels | UPSTREAM | `mmq.cu:109`, `mmvq.cu:1416`, `ggml-cuda.cu:765`, `:1817` | G1 |
| #1170 | bias экспертов в top_k_moe | MoE / CUDA | UPSTREAM | `ggml/src/ggml-cuda/topk-moe.cu:90` | G1 |
| #2389 | `--defer-ple` (ленивая загрузка PLE) | RAM / loader / qwen4exp | UPSTREAM | `TENSOR_READ_LAZY` (`src/llama-model-loader.h:71`), `-lzm` (`common/arg.cpp:2732`), `src/llama-model-loader.cpp:1403`, `:1636` | G2 |
| #1501 | MoE-aware `--fit` | VRAM / fit | UPSTREAM | `common/fit.cpp:547`, `:575`, `:618`, `:729` | G2 |
| #1421 | fusion SSM_CONV+ADD+SILU на CUDA | CUDA / fusion | UPSTREAM (CUDA часть) | `ggml/src/ggml-cuda/ggml-cuda.cu:4078`, `:4083` | G1 |
| #2116 | AVX-VNNI | CPU / AVX | UPSTREAM, уже ON в кэше (`:608`) | `ggml/CMakeLists.txt:156` | G2 |
| #1452 | FA vec D=256 с квантованным KV (фикс offset) | CUDA / FA | UPSTREAM | `fattn-vec.cuh:585`, `fattn.cu:127` | G3 |
| #2345 | DFlash 2 speculative decoding | Speculative decoding | UPSTREAM | `src/models/dflash.cpp` | G3 |
| #1393 | graph reuse | CUDA Graphs | UPSTREAM | `src/llama-context.cpp:1347` | G3 |
| #1094 | graph reuse v2 | CUDA Graphs | UPSTREAM | `src/llama-context.cpp:1347` | G3 |
| #947 | `-gr` graph reuse | CUDA Graphs | UPSTREAM | `src/llama-context.cpp:1347`, `src/llama-context.h:379` | G3 |
| #1702 | включение CUDA graphs | CUDA Graphs | UPSTREAM, форсировано | `CMakeLists.txt:169` | G3 |
| #825 | attention mask tweaks (F16 mask, многопоточный fill) | CUDA / attention mask | UPSTREAM | `src/llama-graph.cpp:38`, `:48`; `GGML_KQ_MASK_PAD` удалён из дерева | G3 |

## 5. BLOCKED - применимо только после другой работы

| ID | Тема | Область | Статус | Эффект | Сложн. | Риск | Группа |
|---|---|---|---|---|---|---|---|
| #1759 | асинхронные копии recurrent-state | TG / delta-net / qwen4exp | BLOCKED | +1..3 % TG, только после #1261 | высокая | высокий | G2, G3 |

## 6. N/A - не применимо (причины в `04-not-applicable.md`)

| ID | Причина | Группа |
|---|---|---|
| #1988 | Linux-only путь (`cudaHostRegister`, `MADV_POPULATE`) | G3 |
| #2310 | чужая архитектура, файл `build_laguna.cpp` отсутствует | G3 |
| #2266 | чужая архитектура (`build_deepseek4`/`openpangu`) | G3 |
| #2347 | CUDA DSA, наш QSA устроен иначе | G3 |
| #928 | ik-специфичная инфраструктура CONCAT+CPY, DS2 граф | G3 |
| #1997 | целевой файл `build_dflash.cpp` отсутствует | G3 |
| #2102 | multi-GPU загрузка | G3 |
| #1942 | FA head 512, у нас head_dim=256 | G3 |
| #1707 | ik-only ядра `ggml_cuda_moe_up_gate_unary` отсутствуют | G3 |
| #1403 | как код: ik-only fused MoE инфраструктура | G3 |
| #1137 | как код: `merge_qkv` отсутствует, 16 файлов через ggml.c/loader | G3 |
| #2339 | IQK бэкенд + RDNA3 | G3 |
| #2107 | IQK файлы; MSVC эффект в пределах шума | G3 |
| #1991 | макросы и кванты ik-специфичны | G3 |
| #2165 | арх deepseek4 | G3 |
| #2144 | sm_60 (P100) | G3 |
| #2065 | арх openPangu, MTP/MLA-latent | G3 |
| #2061 | черновая модель DFlash gpt-oss | G3 |
| #2048 | черновая модель DFlash MiMo | G3 |
| #1921 | head_dim 512 (Gemma 4 12B) | G3 |
| #1788 | vision mmproj, не наш сценарий | G3 |
| #1649 | баг shared memory в delta-net, у нас его нет | G3 |
| #1638 | Volta sm_70 | G3 |
| #1388 | multi-GPU graph parallel | G3 |
| #1283 | ik-путь silu в IQK; у нас свой в `ops.cpp:2626` | G3 |
| #1089 | multi-GPU graph parallel gen2 | G3 |
| #1080 | multi-GPU graph parallel | G3 |
| #1061 | арх Cohere2 | G3 |
| #1022 | CUDA tensor parallel, multi-GPU | G3 |
| #929 | MLA-специфичные FA ядра | G3 |
| #923 | ik-only fused MoE | G3 |
| #868 | fusion в ik-специфичном ggml-cuda | G3 |
| #840 | fusion в ik-специфичном ggml-cuda | G3 |
| #864 | требует ik-only mmvq ядра + ops FUSED_UP_GATE/MOE_FUSED_UP_GATE; перепроверено 2026-09-07 по head `d0861f83f`, в нашем дереве ничего нет | G3 |
| #760 | K-shift это ARM упаковка Q2_K (`arch/arm/quants.c:1412`) | G2 |
| #1278 | нет qwen3next graph-disable, удалять нечего | G1 |
| #1761 | greedy не вызывается при temp=1.0 / top-k 20 | G2 |
| #2010 | Linux-only THP | G2 |
| #1634 | defer experts, неприменимо к нашему лоадеру | G1 |
| #2101 | prefetch experts, неприменимо к нашему лоадеру | G1 |
| #1320 | delta-net CPU путь, у нас delta-net на GPU | G2 |
| #1362 | delta-net CPU путь, у нас delta-net на GPU | G2 |
| #1315 | delta-net CPU путь, у нас delta-net на GPU | G2 |
| #1330 | delta-net CPU путь, у нас delta-net на GPU | G2 |
| #2361 | IQK бэкенд отсутствует | G2 |
| #10 | head128/GQA16 путь, у нас head 256 / gqa 12 | G1 |
| #1221 | head128/GQA16 путь, у нас head 256 / gqa 12 | G1 |
| `6be3a488d` | head128/GQA16 путь, у нас head 256 / gqa 12 | G1 |
