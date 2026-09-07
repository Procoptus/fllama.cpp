# Не применимо - сгруппировано по причине

Каждая группа имеет одну корневую причину. Если причина не изменится, повторно
не проверять ни один PR из группы.

## 1. Нет каталога `ggml/src/iqk/`

В нашем дереве отсутствует IQK-бэкенд ik. Эти оптимизации непереносимы как код.

| ID | Тема | Заявка ik | Примечание |
|---|---|---|---|
| #2361 | IQK кванты | - | каталог отсутствует |
| #2107 | Clang broadcast для IQK | +8.5 % PP (Clang), +20 % IQ4_XS | MSVC-эффект в ik лишь +1.3 % (шум); у нас сборка MSVC |
| #1991 | `HAVE_VNNI256` для IQ4_XS_R8/Qx_0 | - | макросы и кванты ik-специфичны |
| #2339 | IQ4_KS/KT RDNA3 | 16.3 -> 255.9 t/s | плюс нерелевантный бэкенд RDNA3 |
| #929 | DeepSeek FA | ~25 % PP @32k | MLA-специфичные ядра + IQK |
| #1283 | Qwen3-Next CPU silu | +4 % PP | коммит в ik-путях; у нас свой silu `ggml/src/ggml-cpu/ops.cpp:2626` |

## 2. Multi-GPU

Одна RTX 3080. Всё, завязанное на несколько устройств, отпадает.

| ID | Тема | Заявка ik |
|---|---|---|
| #1080 | graph parallel | ~20 % PP |
| #1089 | graph parallel gen2 | ~20 % PP |
| #1388 | Qwen3.5 graph parallel | +25 % |
| #1022 | CUDA tensor parallel | 1.55x PP @26k |
| #2102 | mmap для параллельной загрузки графов | multi-GPU split |

## 3. Чужие архитектуры моделей

У нас `qwen4exp`. PR написаны под другие архитектуры, файлы-цели в дереве
отсутствуют.

| ID | Тема | Архитектура | Проверка |
|---|---|---|---|
| #2310 | `--swa-compress` | laguna | `build_laguna.cpp` отсутствует |
| #2266 | `--swa-compress` | deepseek4 / openpangu | `build_deepseek4`/`openpangu` отсутствуют; `swa_compress` нет, есть только `--swa-full` (`common/arg.cpp:1689`) |
| #2347 | CUDA DSA | deepseek sparse attention | `dsa_attn.cu` отсутствует; наш QSA идёт через `mul_mat`+relu+`ggml_top_k` (`src/models/qwen4exp.cpp:684`) |
| #2165 | DeepSeek4 perf | deepseek4 | +20 % TG @32k |
| #2065 | openPangu MTP / MLA-latent | openPangu | 1.7-2.2x TG; qwen4exp не имеет MTP/nextn |
| #1061 | Cohere2 sm | cohere2 | 600 -> 700 t/s |
| #1921 | FA head_dim 512 | Gemma 4 12B | у нас head_dim=256 |
| #1942 | FA head 512 для Turing | head 512 | у нас head_dim=256 |
| #928 | DeepSeek TG opt (CONCAT+CPY fusion, DS2 graph) | deepseek2 | ik-специфичная инфраструктура |
| #1788 | mmproj n_batch (MTP buffer OOM) | vision | vision mmproj не используется |
| #2061 | DFlash draft gpt-oss | gpt-oss | DFlash уже в дереве, черновая модель не наша |
| #2048 | DFlash draft MiMo | MiMo | то же |
| #1997 | DFlash persistent FA-ready KV | dflash | целевой `build_dflash.cpp` отсутствует, у нас другая реализация |

## 4. Чужая микроархитектура GPU / CPU / ОС

| ID | Тема | Почему нет |
|---|---|---|
| #2144 | P100 FA fp32 | sm_60, у нас sm_86 |
| #1638 | Volta MMQ tile (NO_DEVICE_CODE -> 630 t/s) | sm_70, у нас sm_86 |
| #1988 | faster `ggml_cuda_host_malloc` | Linux-путь (`mmap`+`cudaHostRegister`+`MADV_POPULATE`); у нас Windows, plain `cudaMallocHost` - `ggml/src/ggml-cuda/ggml-cuda.cu:1274` |
| #2010 | Transparent Huge Pages | Linux-only |
| #760 | K-shift | в нашем дереве `k_shift` - только ARM упаковка Q2_K (`ggml/src/ggml-cpu/arch/arm/quants.c:1412`, `:3952`); `k_shift` в `src/llama-kv-cache.cpp:1526` - это RoPE-shift, а не квантовочный |

## 5. ik-only ядра и инфраструктура, которых нет в дереве

`git grep` по именам ядер пуст.

| ID | Тема | Чего нет |
|---|---|---|
| #1707 | small-batch MoE (`moe_up_gate_unary`, `ne[2]<=8`) | `ggml_cuda_moe_up_gate_unary`, `fused_mul_mat_vec_q_id` |
| #864 | fuse up*unary в MMVQ (~2 % TG) | весь PR построен на ik-only инфраструктуре: ops `GGML_OP_FUSED_UP_GATE`/`GGML_OP_MOE_FUSED_UP_GATE`, `ggml_cuda_moe_up_gate_unary`, `ggml_cuda_op_fused_mul_mat_vec_q_id`/`ggml_cuda_op_mul_mat_vec_q_id`, файлы `iqk_mmvq.cu/.cuh/_templates.cuh`, `mmvq-args.h`, `mmvq-templates.cuh` + 50 mmvq template-instance файлов с id-вариантами. Проверено 2026-09-07 на head PR (`d0861f83f`): `git grep` по этим именам в нашем дереве пуст; upstream-`mmvq.cu` не имеет id/fused-вариантов. Портировать можно только переписав всю mmvq/mmid-подсистему с нуля |
| #923 | CUDA MoE (+20-25 % vs baseline) | ik-only `fmoe` |
| #1403 | fused MoE инфраструктура | как код: fused MoE в ik-исполнении |
| #1137 | merged up/gate как код | `merge_qkv` отсутствует; дифф на 16 файлов через `ggml.c`/loader/quantize |
| #868 | fused ops (~1 %) | fusion в ik-специфичном `ggml-cuda` |
| #840 | fused ops (~1 %) | то же |

## 6. Путь не выполняется в нашей конфигурации

| ID | Тема | Почему не выполняется |
|---|---|---|
| #1761 | оптимизация greedy | greedy не вызывается при `temp=1.0`, `top-k=20` |
| #1320 | delta-net CPU путь | дельта-нет полностью на GPU (`ggml/src/ggml-cuda/gated_delta_net.cu`) |
| #1362 | delta-net CPU путь | то же |
| #1315 | delta-net CPU путь | то же |
| #1330 | delta-net CPU путь | то же |
| #1634 | defer experts | неприменимо к нашему лоадеру |
| #2101 | prefetch experts | неприменимо к нашему лоадеру |

## 7. Баг или код, которого у нас нет

| ID | Тема | Проверка |
|---|---|---|
| #1649 | delta-net `__syncthreads` correctness | наш `ggml/src/ggml-cuda/gated_delta_net.cu` shared memory не использует (`grep` пуст) - баг не существует |
| #1278 | убрать graph-disable для qwen3next | такого кода в дереве нет, удалять нечего |

## 8. head_dim / gqa не совпадают

У нас `head_dim = 256`, `gqa_ratio = 12`. Наш случай уже покрыт локальным
коммитом `8377362c7` (`ggml/src/ggml-cuda/fattn.cu:91-96`).

| ID | Тема |
|---|---|
| #10 | FA путь под head128 / GQA16 |
| #1221 | FA путь под head128 / GQA16 |
| `6be3a488d` | FA путь под head128 / GQA10 |
