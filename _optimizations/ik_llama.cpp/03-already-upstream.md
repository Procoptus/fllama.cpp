# Эквивалент уже есть в upstream - повторно не проверять

Формат: ID | что делает ik | где наш эквивалент | группа проверки

| ID | Что делает | Наш эквивалент | Группа |
|---|---|---|---|
| #2316 | uid графа + сравнение адресов src для решения о повторном capture | `ggml_graph_next_uid()` - `ggml/src/ggml.c:56`; поле `uid` - `ggml/src/ggml-impl.h:345`; `ggml/src/ggml-backend.cpp:1085`. Коммиты upstream `3f7c29d31` (#21764), `d12cc3d1c` (#21635) | G1 |
| #2136 | повторный capture при CPY | `ggml_cuda_graph_update_required` - `ggml/src/ggml-cuda/ggml-cuda.cu:2624-2664` (memcmp по всем `node_src_data_ptrs`, без исключения для CPY) | G1 |
| #2292 | трёхуровневая защита padding | `ggml/src/ggml-cuda/mmq.cu:109-118`; `ggml/src/ggml-cuda/mmvq.cu:1416-1423`; `ggml/src/ggml-cuda/ggml-cuda.cu:765-774`, `:1863-1866` (`bad_padding_clear`) | G1 |
| #1170 | bias экспертов учитывается в top_k_moe | `template <int n_experts, bool has_bias>` - `ggml/src/ggml-cuda/topk-moe.cu:90` | G1 |
| #2389 | `--defer-ple`, ленивая загрузка per_layer_token_embd | `TENSOR_READ_LAZY` - `src/llama-model-loader.h:71`; `-lzm/--lazy-mode` - `common/arg.cpp:2732`, `:2737`; `llama_lazy_mode` - `include/llama.h:217`; применение в `src/models/qwen4exp.cpp:187`. Конфликт mlock x lazy уже решён: `src/llama-model-loader.cpp:1403` и `:1636` («locking a lazy tensor would fault all of it in») | G2 |
| #1501 | MoE-aware `--fit` | `common/fit.cpp:547`, `:575` (`LAYER_FRACTION_MOE`), `:618` (`pattern_moe_all`), `:729` (`global_surplus_cpu_moe`) | G2 |
| #1421 | fusion SSM_CONV+ADD+SILU / SSM_CONV+SILU | `ggml/src/ggml-cuda/ggml-cuda.cu:4078` (`{SSM_CONV, ADD, SILU}`), `:4083` (`{SSM_CONV, SILU}`). CUDA часть применима, CPU часть N/A | G1 |
| #2116 | AVX-VNNI | уже `ON` в `build_avx_512/CMakeCache.txt:608`; опция `ggml/CMakeLists.txt:156` | G2 |
| #1452 | FA vec D=256 с квантованным KV (фикс offset в `quantize_q8_1_to_shared`) | `ggml/src/ggml-cuda/fattn-vec.cuh:585` (D=256), `ggml/src/ggml-cuda/fattn.cu:127-187` (dispatch по head size), `ggml/src/ggml-cuda/fattn-common.cuh:641` | G3 |
| #2345 | DFlash 2 speculative decoding (selector/conv тензоры) | `src/models/dflash.cpp` - полная реализация DFlash2/DSpark. Дифф ik таргетит отсутствующий `src/graphs/build_dflash.cpp` | G3 |
| #1393 | graph reuse | `res->can_reuse(gparams)` - `src/llama-context.cpp:1347` | G3 |
| #1094 | graph reuse v2 | `src/llama-context.cpp:1347` | G3 |
| #947 | `-gr` graph reuse (prev-graph + cache_copies view_offs) | `graph_reuse_disable` / `can_reuse` - `src/llama-context.cpp:1347`, `src/llama-context.h:379`. Ручная мутация `view_offs` из ik непереносима в нашу архитектуру kv-cache | G3 |
| #1702 | включение CUDA graphs | форсировано `CMakeLists.txt:169`; `GGML_CUDA_GRAPHS=ON` в кэше `:683` | G3 |
| #825 | attention mask tweaks: F16 mask + многопоточный fill (2-4 % на длинном контексте) | F16 mask уже безусловный - `src/llama-graph.cpp:38`; `GGML_KQ_MASK_PAD` удалён из дерева; многопоточный fill покрыт `can_reuse_kq_mask` - `src/llama-graph.cpp:48` | G3 |

## Дополнительно закрыто, но не как «перенос»

- `#1599` - механизм квантованного KV в upstream есть, но требует параметра
  запуска. См. `02-applicable.md`, приоритет 2.
- `#1261` - ngram self-speculation в upstream есть, требует параметра. См.
  `02-applicable.md`, приоритет 3.
- `#2262` - `-ncmoe` в upstream есть. См. `02-applicable.md`, раздел рычагов.
- `#1137`/`#1403` - merged gate/up в upstream есть, но GGUF может не использовать.
  См. `02-applicable.md`, приоритет 4.
- `#1560` - механизм `n_outputs_max` в upstream есть, нет только CLI. См.
  `02-applicable.md`, приоритет 9.

## Локальные порты (уже в нашем дереве)

| Локальный коммит | Соответствует ik | Где |
|---|---|---|
| `c84ce61fc` | #2297 | FA |
| `8377362c7` | #2372 | `ggml/src/ggml-cuda/fattn.cu:91-96` - `DKQ==256 && gqa_ratio==12 -> ncols2=4` |
| `ef5ac8948` | #2373 | `ggml/src/ggml-cuda/ggml-cuda.cu:1817` (`k_simple_gemm_f32`), `:1840` (`ggml_cuda_mul_mat()` fast path) |
