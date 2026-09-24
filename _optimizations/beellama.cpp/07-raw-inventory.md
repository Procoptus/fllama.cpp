# Сырой инвентарь коммитов: beellama.cpp (Anbeeld)

Дата сбора: 2026-09-03
Репозиторий: https://github.com/Anbeeld/beellama.cpp
Метод: git clone --filter=blob:none + git log <fork-HEAD> --not upstream/master

## Параметры выборки
| Параметр | Значение |
|---|---|
| Ветка форка (HEAD) | main |
| Форк HEAD | `cd3c41e7` = `cd3c41e73d0a5deb1d2051bec04df73673a40e9c` |
| Дата/subject HEAD | 2026-09-01 01:52:35 +0200 - ci: bypass registry cache for ROCm image |
| merge-base с upstream/master | `6fdd0ac89` = `6fdd0ac8907fd973a42b876357823ad2124cd8ed` |
| Дата merge-base | 2026-08-27 (ci : bundle HIP runtime DLLs with Windows ROCm release #26973) |
| Ahead (fork commits вне upstream) | 915 |
| Behind (upstream-коммитов не хватает форку) | 132 |
| Не-merge коммитов вне upstream | 848 |
| Ref для сравнения | upstream/master = 95ef7fc16054e63b427a3ef00188e055ef7586d8 (2026-09-03) |


## Архитектурная совместимость форка

| маркер | найден в форке? (да/нет, файл:строка) |
|---|---|
| `qwen4exp` | no | - |
| `LLM_ARCH_QWEN4EXP` | no | - |
| `per_layer_token_embd` | yes | conversion/base.py:978; gguf-py/gguf/constants.py:1418; gguf-py/gguf/constants.py:689 |
| `ffn_gate_up_exps` | yes | gguf-py/gguf/constants.py:1411; src/llama-arch.cpp:433; src/llama-arch.cpp:848 |
| `ffn_gate_inp_shexp` | yes | conversion/base.py:945; gguf-py/gguf/constants.py:1390; gguf-py/gguf/constants.py:659 |
| `build_attn_qsa` | no | - |
| `ggml_ssm_conv` | yes | ggml/include/ggml.h:2566; ggml/src/ggml-vulkan/ggml-vulkan.cpp:21829; ggml/src/ggml.c:5762; ggml/src/ggml.c:5764 |
| `causal_conv1d` | yes | src/models/bailingmoe3.cpp:169; src/models/bailingmoe3.cpp:263; src/models/bailingmoe3.cpp:266; src/models/bailingmoe3.cpp:269 |
| `mul_mat_id` | yes | ggml/include/ggml.h:1476; ggml/include/ggml.h:538; ggml/src/ggml-backend-meta.cpp:1021; ggml/src/ggml-backend.cpp:1741 |
| `MMID` | yes | ggml/src/ggml-cpu/ggml-cpu.c:1492; ggml/src/ggml-cpu/ggml-cpu.c:1532; ggml/src/ggml-cpu/ggml-cpu.c:1669; ggml/src/ggml-cpu/repack.cpp:4449 |
| `gated_delta` | yes | ggml/include/ggml.h:2681; ggml/include/ggml.h:2694; ggml/include/ggml.h:596; ggml/src/ggml-backend-meta.cpp:872 |
| `delta_net` | yes | ggml/include/ggml.h:2681; ggml/include/ggml.h:2694; ggml/include/ggml.h:596; ggml/src/ggml-backend-meta.cpp:872 |
| `chunked_prefill` | no | - |
| `KVP` | no | - |
| `GGML_CUDA_FA` | yes | ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:50; ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu:56 |

**Модель qwen4exp:**
- qwen4exp model: NOT supported. No src/models/qwen4exp.cpp, no conversion/qwen4exp.py.
- qwen* models in src/models: qwen, qwen2, qwen2moe, qwen2vl, qwen3, qwen35, qwen35moe,
- qwen3moe, qwen3next, qwen3tts, qwen3vl, qwen3vlmoe, rwkv6qwen2 - no qwen4exp.
- beellama focus is different: ROCm/HIP, DFlash speculative decoding, 'kvarn' FA backend,
- Windows CI/release packaging - not the qwen4exp/QSA architecture.

**Пересечение с нашими 24 локальными коммитами (те же темы):**
- rms_norm_add / fuse rms_norm: ABSENT (no fused rms_norm_add op found in code).
- ncols2: PRESENT in ggml/src/ggml-cuda/fattn-common.cuh (FA tile-col machinery), but no
- explicit 'head 256'/'FA head256'/'mma f16 head 256' marker found.
- small F32 GEMM: ABSENT (no small_f32_gemm symbol).
- QSA top_k skip: ABSENT (no qsa/top_k_skip - beellama has no QSA at all).
- iq_rows / IQP AVX512 VNNI Q5_K/Q6_K: ABSENT (no iqp/iq_rows symbols; VNNI/Q5_K/Q6_K
- exist as generic upstream quant types, not a fork-specific IQP AVX512 VNNI pass).
- Separate FA backend found: ggml/src/ggml-cuda/fattn-kvarn-dispatch.cu (GGML_CUDA_FA).

| тема (наш локальный 24) | статус в форке | доказательство |
|---|---|---|
| `rms_norm_add` | no | - |
| `ncols2` | yes | ggml/src/ggml-cuda/fattn-common.cuh:1259; ggml/src/ggml-cuda/fattn-common.cuh:1272; ggml/src/ggml-cuda/fattn-common.cuh:1281; ggml/src/ggml-cuda/fattn-common.cuh:1300 |
| `head256` | no | - |
| `head` | no | - |
| `FA_HEAD256` | no | - |
| `small_f32_gemm` | no | - |
| `small` | no | - |
| `top_k_skip` | no | - |
| `qsa` | no | - |
| `iq_rows` | no | - |
| `VNNI` | yes | ggml/src/ggml-cpu/ggml-cpu.c:3675; ggml/src/ggml-cpu/ggml-cpu.c:3707; ggml/src/ggml-cpu/ggml-cpu.cpp:46; ggml/src/ggml-cpu/ggml-cpu.cpp:613 |
| `Q5_K` | yes | convert_llama_ggml_to_gguf.py:42; convert_llama_ggml_to_gguf.py:43; ggml/include/ggml.h:403; ggml/include/ggml.h:486 |
| `Q6_K` | yes | conversion/base.py:989; convert_llama_ggml_to_gguf.py:44; ggml/include/ggml.h:404; ggml/include/ggml.h:487 |
| `iqp` | no | - |

## Полный список коммитов форка вне upstream

Всего не-merge коммитов: **848** (> 400). По правилу: (a) общее число = 848; (b) сгруппированная разбивка ниже (AREA / SUBJECT PREFIX); (c) ПОЛНЫЙ перечень коммитов с перф-маркерами = 328 строк; (d) top-120 остальных по числу изменённых файлов. Полный список всех 848 строк сохранён во временном файле _forks_opt/beellama_classified.md (раздел FULL LIST).

### (b) Разбивка по областям (top directory) и по префиксу subject
| Область (префикс пути) | Кол-во коммитов | Примеры SHA |
|---|---|---|
| src/ | 152 | c6c438f4e 11f45ed34 2474373ec 5ecbe1ac1 |
| ggml/src/ggml-cuda | 149 | c7a6b18d4 d1a522fc8 7ea40ee98 fde7797a2 |
| other | 125 | 3746fb185 fb500579d 005128c49 287da7cd0 |
| tests/ | 90 | 2a5e1bc9e a7417e85b 2d7a40c6b 1ce739a01 |
| common/ | 90 | f5a7ec15d c184d9a7a b8a4a8532 c314bb10d |
| .github | 54 | cd3c41e73 ed8db4abf 5a357925e e66f4d4f2 |
| src/models | 39 | 2f3923bc8 f7aadef09 cf095c834 bfaa12563 |
| docs | 39 | 0fb10f6b4 9084f1bed 5eaba174c 41d41b0ce |
| tools/server | 33 | fc4c83995 fc9ab2bd6 83d4173d8 8fb188a6f |
| ggml/src/ggml-metal | 16 | 6d019c3b8 413b33253 aa6a3a180 654647aac |
| ggml/src/other | 15 | 2525c60cf 634ffc9be 83429d46a 9572674dc |
| ggml/src/ggml-cpu | 11 | ad33bb4b2 e4cfc12f8 0ef8c42f4 9f268fb03 |
| tools/other | 9 | ab8a22e5b 8985e86cb c5eefecf8 4db14be0a |
| examples/ | 8 | 1006e84f0 0e4945217 edecf9699 d16a50d3b |
| scripts/ | 6 | e9eb7eec1 bc4558e60 4a834dd7a 1ae630920 |
| ggml/src/ggml-vulkan | 4 | 2c919f2ce a620cbd48 b639edb1d 12a169b8f |
| gguf-py | 3 | 64f765f5a 64885bc9d 9d4a73a49 |
| cmake | 2 | d237e4e47 89364abbe |
| ggml/include | 2 | e1cf4bf52 7cae8affd |
| conversion/ | 1 | e0181cd6c |

| Префикс subject | Кол-во |
|---|---|
| (long) | 232 |
| dflash | 72 |
| fix | 51 |
| ci | 49 |
| docs | 48 |
| server | 39 |
| feat | 34 |
| cuda | 23 |
| perf | 18 |
| Update CHANGELOG.md | 16 |
| Update README.md | 11 |
| kvarn | 9 |
| common | 9 |
| cont | 8 |
| build | 7 |
| CUDA | 7 |
| fix(kvarn) | 7 |
| tests | 6 |
| fix(ci) | 6 |
| DFlash | 5 |
| ggml-cuda | 5 |
| B2.2 | 5 |
| hip | 4 |
| kv-cache | 4 |
| ggml-cuda(kvarn) | 4 |
| models | 4 |
| ggml | 4 |
| v0.2.0 | 4 |
| llama | 4 |
| feat(dflash) | 4 |
| SD-076 | 4 |
| C.0 | 4 |
| fix(server) | 3 |
| fix(cuda) | 3 |
| cmake | 3 |
| test(dflash-plumbing) | 3 |
| feat(gemma4-iswa) | 3 |
| vocab | 3 |
| simplify | 3 |
| docker | 2 |

### (c) Коммиты с перф-маркерами (полный перечень, 328)
| SHA | Дата | Автор | Subject | Изменено файлов | Ключевые директории | Перф-маркеры |
|---|---|---|---|---|---|---|
| `cd3c41e73` | 2026-09-01 | Anbeeld | ci: bypass registry cache for ROCm image | 2 | .github(1), tests/(1) | cache |
| `fb500579d` | 2026-08-29 | Anbeeld | fix Windows CUDA build scripts | 5 | other(2), scripts/(2), tests/(1) | cuda |
| `c7a6b18d4` | 2026-08-27 | Anbeeld | Fix KV checkpoint restore and KVarN stage ownership | 23 | ggml/src/ggml-cuda(7), src/(5), tests/(4) | kv |
| `11f45ed34` | 2026-08-26 | SubSir | Fix graph number calculation | 1 | src/(1) | graph |
| `7ea40ee98` | 2026-08-22 | SubSir | Optimize Dflash 2 cost | 3 | ggml/src/ggml-cuda(1), src/models(1), tests/(1) | optimize |
| `0fce08dfb` | 2026-08-20 | Anbeeld | fix(mtp): synchronize multi-ubatch draft decode | 3 | tests/(2), src/(1) | ubatch,decode |
| `fde7797a2` | 2026-08-15 | Anbeeld | Optimize native KVarN SWA attention paths | 20 | ggml/src/ggml-cuda(9), src/(5), tests/(4) | optimize |
| `856598ad1` | 2026-08-12 | Anbeeld | Fix unified KVarN cache capacity sharing | 21 | ggml/src/ggml-cuda(7), src/(5), tests/(3) | cache,unified |
| `973729625` | 2026-08-12 | Anbeeld | docs: clarify prompt cache prefill improvement | 1 | other(1) | improvement,cache,prefill,prompt |
| `bc4433363` | 2026-08-12 | Anbeeld | kvarn: restore multi-slot prefill and checkpoint throughput | 22 | src/(9), ggml/src/ggml-cuda(7), ggml/src/ggml-vulkan(3) | prefill |
| `fc9ab2bd6` | 2026-08-11 | Anbeeld | server: remediate prompt cache restore transactions | 19 | tools/server(7), tests/(4), src/(3) | cache,prompt |
| `16c76b14a` | 2026-08-11 | Anbeeld | server: make prompt cache state selective and transactional | 35 | src/(19), tools/server(5), common/(4) | cache,prompt |
| `2e96c52ea` | 2026-08-09 | Anbeeld | Fix portable KVarN attention on pre-Turing CUDA | 5 | ggml/src/ggml-cuda(2), tests/(2), docs(1) | cuda |
| `32040e79d` | 2026-08-09 | Anbeeld | fix: keep KVarN AMD decode on supported routes | 6 | ggml/src/ggml-cuda(2), tests/(2), docs(1) | decode |
| `860d918ff` | 2026-08-06 | Anbeeld | fix(cuda): set KVarN decode combine shared-mem limit once, cover VEC path | 2 | ggml/src/ggml-cuda(2) | cuda,decode |
| `695a3f485` | 2026-08-02 | Developer | fix(cuda): opt KVarN decode combine kernel into larger dynamic shared mem | 1 | ggml/src/ggml-cuda(1) | opt,kernel,cuda,decode |
| `9d59db2e3` | 2026-07-29 | Anbeeld | ci: remove exhaustive CUDA architecture release gate | 5 | .github(2), tests/(2), docs(1) | cuda |
| `4da53cce4` | 2026-07-29 | Anbeeld | ci: remove exhaustive CUDA architecture release gate | 5 | .github(2), tests/(2), docs(1) | cuda |
| `c4bae88e0` | 2026-07-29 | crusaderky | cuda: silence KVarN MMA compilation warnings | 1 | ggml/src/ggml-cuda(1) | cuda,mma |
| `4a834dd7a` | 2026-07-28 | Anbeeld | Update build-win-cuda-13.1-sm_86.ps1 | 1 | scripts/(1) | cuda |
| `c11000ebf` | 2026-07-28 | Anbeeld | Enable portable KVarN attention on pre-Turing CUDA | 16 | ggml/src/ggml-cuda(5), docs(3), tests/(3) | cuda |
| `d36a9bf7a` | 2026-07-28 | Anbeeld | Fix CUDA Q2_0 template declarations | 3 | ggml/src/ggml-cuda(2), tests/(1) | cuda |
| `66c1f9481` | 2026-07-28 | Anbeeld | Fix CUDA RHS alignment and SWA KVarN rollback | 11 | ggml/src/ggml-cuda(5), tests/(4), src/(2) | cuda |
| `1f06a2dfb` | 2026-07-27 | Andgihat | CUDA: add Q2_0 (ternary, group-64) weight support | 17 | ggml/src/ggml-cuda(17) | cuda |
| `86ec665fa` | 2026-07-27 | Anbeeld | cuda: validate RHS strides for MMVF and MMF | 6 | ggml/src/ggml-cuda(5), tests/(1) | cuda |
| `2c919f2ce` | 2026-07-26 | Anbeeld | Optimize Vulkan KVarN execution and telemetry | 15 | ggml/src/ggml-vulkan(7), src/(3), tests/(2) | optimize |
| `e289bb8e9` | 2026-07-25 | Anbeeld | kv-cache: fix compact SWA tail correctness and memory | 31 | src/(17), docs(3), tests/(3) | cache,kv,memory |
| `cc71513ee` | 2026-07-24 | Anbeeld | kv-cache: compact SWA precision-tail storage | 52 | src/(25), tests/(6), docs(5) | cache,kv |
| `ca155ad07` | 2026-07-22 | Anbeeld | kvarn: fix portable prefill and Vulkan decode | 11 | src/(5), ggml/src/ggml-vulkan(3), other(1) | decode,prefill |
| `89aeb4821` | 2026-07-22 | Anbeeld | cuda: fix KVarN speculative decode scaling | 14 | ggml/src/ggml-cuda(10), tests/(2), other(1) | cuda,decode |
| `4ff1af58e` | 2026-07-20 | Anbeeld | cuda: keep speculative KVarN batches on native split decode | 12 | src/(5), ggml/src/ggml-cuda(4), tests/(2) | cuda,decode |
| `e6b733ee0` | 2026-07-19 | Anbeeld | kvarn: batch exact-tail checkpoint transfers | 6 | src/(3), tests/(2), other(1) | batch |
| `5eaba174c` | 2026-07-19 | Anbeeld | cuda: reduce default FA pair matrix | 14 | docs(5), other(3), ggml/src/ggml-cuda(2) | cuda,fa |
| `83f0e4b0d` | 2026-07-19 | Anbeeld | tests: cover serving-size exact KV tails | 2 | tests/(2) | kv |
| `3fbecf900` | 2026-07-18 | Anbeeld | dflash: rotate injected quantized KV cache | 2 | src/models(1), tests/(1) | cache,kv |
| `4c0aa17c1` | 2026-07-17 | Anbeeld | Fix llama -Werror warnings in KV tail and graph code | 6 | src/(6) | kv,graph |
| `ad33bb4b2` | 2026-07-17 | Anbeeld | Handle BeeLlama quant types in clamp forward switch | 1 | ggml/src/ggml-cpu(1) | quant |
| `e4cfc12f8` | 2026-07-17 | Anbeeld | Fix release build errors across CPU, CUDA, SYCL and Vulkan | 4 | ggml/src/ggml-cpu(2), ggml/src/other(1), ggml/src/ggml-vulkan(1) | cuda,cpu |
| `f4beb51ec` | 2026-07-16 | Anbeeld | Restore staged KVarN eager prefill stores | 5 | tests/(3), docs(1), ggml/src/ggml-cuda(1) | prefill |
| `8fb188a6f` | 2026-07-16 | Anbeeld | Align KV rollback to multimodal chunk boundaries | 4 | tools/server(3), tests/(1) | kv |
| `a51b763fd` | 2026-07-16 | Anbeeld | Unify KVarN CUDA compilation toggle | 14 | other(3), docs(3), ggml/src/ggml-cuda(3) | cuda |
| `9c3d0b489` | 2026-07-16 | Anbeeld | Add provenance-safe paired KV tail benchmarks | 2 | other(2) | kv |
| `86f5f0491` | 2026-07-16 | Anbeeld | Make prompt-cache tail reuse atomic and ownership-safe | 3 | tools/server(3) | cache,prompt |
| `a492265d6` | 2026-07-16 | Anbeeld | Make exact KV tails capability-driven and transactional | 32 | src/(25), src/models(3), common/(2) | kv |
| `5ffaf4f23` | 2026-07-15 | Anbeeld | Restore metadata-capable KVarN fast decode | 26 | src/(14), ggml/src/ggml-cuda(8), tests/(2) | decode |
| `44ac8f800` | 2026-07-15 | Anbeeld | Select exact-tail defaults by cache family | 9 | docs(4), common/(2), other(1) | cache |
| `222688561` | 2026-07-14 | Anbeeld | Document exact standard KV tail representations | 2 | docs(2) | kv |
| `83429d46a` | 2026-07-14 | Anbeeld | Complete standard quantized KV tail backend operations | 33 | ggml/src/other(13), ggml/src/ggml-vulkan(8), ggml/src/ggml-metal(7) | kv |
| `9f5850a04` | 2026-07-14 | Anbeeld | Implement exact standard KV tail storage planning | 24 | src/(17), tests/(5), common/(2) | kv |
| `a0a884b99` | 2026-07-14 | Anbeeld | Optimize fully covered exact KV groups | 6 | src/(4), tests/(2) | optimize,kv |
| `8f460db86` | 2026-07-14 | Anbeeld | Explain standard-cache exact-tail behavior | 4 | docs(4) | cache |
| `52c9acec6` | 2026-07-14 | Anbeeld | Implement exact tails for quantized standard KV caches | 42 | src/(23), tests/(7), common/(4) | kv |
| `037572762` | 2026-07-11 | Anbeeld | Restore low-bit CUDA KV cache writes | 10 | ggml/src/ggml-cuda(3), other(2), tests/(2) | cuda,cache,kv |
| `888791e1e` | 2026-07-11 | Anbeeld | Document KVarN ubatch guidance | 3 | docs(3) | ubatch |
| `31d87effb` | 2026-07-11 | Anbeeld | Forward-port late v0.3.2 CUDA fixes | 4 | ggml/src/ggml-cuda(2), ggml/CMakeLists.txt(1), tests/(1) | cuda |
| `a12d7e228` | 2026-07-09 | Anbeeld | cmake: limit default KVarN fast decode pairs | 7 | ggml/src/ggml-cuda(2), ggml/src/other(2), docs(1) | decode |
| `f6ffda723` | 2026-07-09 | Anbeeld | Revert "cmake: drop CUDA 12 Maxwell PTX default" | 1 | ggml/src/ggml-cuda(1) | cuda |
| `b27d9ac93` | 2026-07-08 | Anbeeld | cmake: drop CUDA 12 Maxwell PTX default | 1 | ggml/src/ggml-cuda(1) | cuda |
| `9572674dc` | 2026-07-04 | Anbeeld | Preserve hidden KVarN view sources in CUDA scheduler | 1 | ggml/src/other(1) | cuda |
| `476d37e1b` | 2026-07-03 | Anbeeld | Add explicit KVarN SWA cache-type overrides | 11 | src/(5), tests/(3), common/(2) | cache |
| `d5c661b07` | 2026-06-30 | Anbeeld | Enable Gemma windowed KVarN prefill | 3 | ggml/src/ggml-cuda(1), src/(1), tests/(1) | prefill |
| `8e5cd8318` | 2026-06-30 | Anbeeld | Add faithful windowed KVarN prefill | 4 | tests/(2), ggml/src/ggml-cuda(1), src/(1) | prefill |
| `d7f2cad76` | 2026-06-29 | Anbeeld | Optimize KVarN store serialization | 1 | ggml/src/ggml-cuda(1) | optimize |
| `5eb563c04` | 2026-06-29 | Anbeeld | Clean up KVarN prefill precision changes | 4 | ggml/src/ggml-cuda(2), tests/(2) | prefill |
| `0b6ee0a6b` | 2026-06-28 | Anbeeld | CUDA: shard KVarN MMA instantiations | 22 | ggml/src/ggml-cuda(22) | cuda,mma |
| `c6ab02cd9` | 2026-06-25 | Anbeeld | ggml-cuda(kvarn): hoist dual-axis K dequant out of decode MMA inner loop | 1 | ggml/src/ggml-cuda(1) | cuda,mma,decode |
| `c34087d3c` | 2026-06-25 | Anbeeld | ggml-cuda(kvarn): widen vec decode to 4 warps / 128 threads | 1 | ggml/src/ggml-cuda(1) | cuda,decode |
| `7fac7dc00` | 2026-06-24 | Anbeeld | ggml-cuda(kvarn): generalize vec decode bit pairs | 40 | ggml/src/ggml-cuda(38), tests/(2) | cuda,decode |
| `9248bcaeb` | 2026-06-24 | Anbeeld | ggml-cuda(kvarn): geometry-driven decode selection, drop F16 materialization | 10 | ggml/src/ggml-cuda(8), tests/(2) | cuda,decode |
| `098e00d34` | 2026-06-23 | Anbeeld | Generalize KVarN native decode and parallelize CUDA builds | 45 | ggml/src/ggml-cuda(43), tests/(2) | cuda,decode |
| `555c78c0f` | 2026-06-22 | Anbeeld | Fix KVarN native decode scaling | 6 | ggml/src/ggml-cuda(4), tests/(2) | decode |
| `de3873f01` | 2026-06-21 | Anbeeld | Speed up KVarN native decode at depth | 4 | ggml/src/ggml-cuda(3), tests/(1) | speed,decode |
| `60c446013` | 2026-06-21 | Anbeeld | Speed up KVarN MMA decode loader | 3 | ggml/src/ggml-cuda(3) | speed,mma,decode |
| `8b4f10694` | 2026-06-20 | Anbeeld | Fix KVarN deep native decode path | 2 | ggml/src/ggml-cuda(1), tests/(1) | decode |
| `aae98fb1c` | 2026-06-20 | Anbeeld | Fix KVarN large-prefill routing | 14 | src/(8), ggml/src/ggml-cuda(3), tests/(2) | prefill |
| `a34f11134` | 2026-06-20 | Anbeeld | Remove graph-level KVarN materialization | 23 | ggml/src/ggml-cuda(4), other(3), ggml/src/ggml-cpu(3) | graph |
| `a8061cc9c` | 2026-06-19 | Anbeeld | Add native KVarN MMA FlashAttention | 4 | ggml/src/ggml-cuda(3), tests/(1) | mma |
| `9dfe9634f` | 2026-06-18 | Anbeeld | Make KVarN staging independent of physical ubatch | 12 | src/(4), ggml/src/ggml-vulkan(3), tests/(2) | ubatch |
| `a620cbd48` | 2026-06-17 | Anbeeld | Fix SWA KVarN audit findings: OOB shared memory, force-materialize crash, ring sizing, Vulkan | 6 | ggml/src/ggml-vulkan(3), src/(2), ggml/src/ggml-cuda(1) | memory |
| `189fdf4c1` | 2026-06-17 | Anbeeld | Remove dead generic KVarN materialize CUDA kernel | 1 | ggml/src/ggml-cuda(1) | kernel,cuda |
| `82933d90f` | 2026-06-16 | Anbeeld | Add opt-in rotated-domain attention for KVarN decode | 9 | src/(4), ggml/src/ggml-cuda(2), tests/(2) | opt,decode |
| `21dd5bb4f` | 2026-06-16 | Anbeeld | Add native CUDA FlashAttention path for KVarN cache | 6 | ggml/src/ggml-cuda(4), tests/(2) | cuda,cache |
| `3ea75b141` | 2026-06-16 | Anbeeld | Add experimental KVarN fused attention | 5 | ggml/src/ggml-cuda(3), tests/(2) | fused |
| `c8da45a37` | 2026-06-16 | Anbeeld | Fix prompt-cache host budget enforcement | 5 | tools/server(3), tests/(2) | cache,prompt |
| `8b250fba2` | 2026-06-15 | Anbeeld | Fix HIP FA launch bounds and Windows CPU packaging | 4 | .github(1), ggml/src/ggml-cuda(1), scripts/(1) | fa,cpu |
| `df8933c26` | 2026-06-15 | Anbeeld | Allow ignoring uncompiled CUDA FA pairs | 6 | tests/(2), other(1), docs(1) | cuda,fa |
| `0776d1da9` | 2026-06-15 | Anbeeld | Fix issue 41 TurboQuant CUDA paths | 6 | ggml/src/ggml-cuda(2), tests/(2), conversion/(1) | cuda |
| `2a486938e` | 2026-06-15 | Anbeeld | Fix CUDA FA vec decode regression | 3 | tests/(2), ggml/src/ggml-cuda(1) | cuda,fa,decode |
| `6fa65bd79` | 2026-06-14 | Anbeeld | Fix q8 Turbo V CUDA FA decode routing | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda,fa,decode |
| `100fc138f` | 2026-06-14 | Anbeeld | Fix DFlash mixed spec rollback planning | 5 | common/(2), tests/(2), tools/server(1) | spec |
| `decf36a36` | 2026-06-14 | Anbeeld | cuda: report missing FA quant pairs | 12 | other(3), docs(3), ggml/src/ggml-cuda(3) | cuda,fa,quant |
| `20dc117dc` | 2026-06-13 | Anbeeld | cuda: treat turbo as q1 half FA rank | 9 | other(3), ggml/src/ggml-cuda(3), docs(1) | cuda,fa |
| `7c2c763e4` | 2026-06-13 | Anbeeld | docs: sync CLAUDE.md/AGENTS.md with v0.3.2 cache types and KVarN | 2 | other(2) | cache |
| `8de599723` | 2026-06-13 | Anbeeld | docs: recommend default FA quant builds | 8 | other(3), docs(3), .github(1) | fa,quant |
| `e3cce5749` | 2026-06-13 | Anbeeld | cuda: make default FA pairs q-cache focused | 12 | other(3), docs(3), ggml/src/ggml-cuda(2) | cuda,fa,cache |
| `cb83fe74f` | 2026-06-13 | Anbeeld | cuda: fix Turbo K classic V FA prefill | 4 | tests/(3), ggml/src/ggml-cuda(1) | cuda,fa,prefill |
| `60f592641` | 2026-06-13 | Anbeeld | Enable unified KV for KVarN | 20 | src/(11), tools/server(4), tests/(3) | kv,unified |
| `52740a995` | 2026-06-13 | Anbeeld | Optimize KVarN materialize hot paths | 3 | tests/(2), ggml/src/ggml-cuda(1) | optimize |
| `13d15a375` | 2026-06-12 | Anbeeld | Fix KVarN graph-reuse correctness | 5 | tests/(2), ggml/src/ggml-cuda(1), ggml/src/other(1) | graph |
| `4b9984d1a` | 2026-06-12 | Anbeeld | Optimize KVarN CUDA store and materialize | 10 | ggml/src/ggml-cuda(5), src/(2), tests/(2) | optimize,cuda |
| `bc304245c` | 2026-06-11 | Anbeeld | Use q2/q3 KVarN fallback cache types | 6 | tests/(2), other(1), common/(1) | cache |
| `f0e5f8ff6` | 2026-06-11 | Anbeeld | Add turbo4 TCQ and expanded KV cache quants | 236 | ggml/src/ggml-cuda(216), ggml/src/other(6), ggml/src/ggml-cpu(4) | cache,kv |
| `e04bde37e` | 2026-06-10 | Anbeeld | kvarn: derive non-KVarN layer cache types from KVarN bit width | 6 | other(1), common/(1), docs(1) | cache |
| `95139a777` | 2026-06-09 | Anbeeld | cuda: fix D512 mixed Turbo FA routing | 1 | ggml/src/ggml-cuda(1) | cuda,fa |
| `2f403a08e` | 2026-06-09 | Anbeeld | cuda: route Gemma-sized mixed Turbo FA away from vec | 1 | ggml/src/ggml-cuda(1) | cuda,fa |
| `e623d3984` | 2026-06-09 | Anbeeld | cuda: add f16 fallbacks to half FA quant builds | 14 | other(4), docs(3), ggml/src/ggml-cuda(2) | cuda,fa,quant |
| `a596a5c72` | 2026-06-07 | Anbeeld | feat(cuda-fa): add HALF_QUANTS build mode for FA vec K/V pairs | 96 | ggml/src/ggml-cuda(86), other(3), docs(3) | cuda,fa |
| `9f1bcc966` | 2026-06-07 | Anbeeld | fix(kvarn): cast HIP kernel pointer for dynamic shmem attribute | 1 | ggml/src/ggml-cuda(1) | kernel |
| `7eff066f2` | 2026-06-06 | Anbeeld | ci: add -j to windows-cpu build, shorten docker job names | 1 | .github(1) | cpu |
| `13ad92aaa` | 2026-06-06 | Anbeeld | server: stop deriving checkpoint budget from cache-ram | 4 | tools/server(3), tests/(1) | cache |
| `5ec9c500c` | 2026-06-06 | Anbeeld | feat(kvarn): group-range state serialization for prompt-cache compression | 2 | src/(1), tests/(1) | cache,prompt |
| `c9ad13d19` | 2026-06-05 | Anbeeld | Add full KVarN 5/6/8 cache pair support | 11 | other(2), tests/(2), common/(1) | cache |
| `b17c380c1` | 2026-06-05 | Anbeeld | Fix prompt checkpoint threshold for multimodal reuse | 3 | tests/(2), tools/server(1) | prompt |
| `353b890a0` | 2026-06-05 | Anbeeld | Fix MTP prompt-cache reuse without regular checkpoints | 5 | tools/server(3), tests/(2) | cache,prompt |
| `819136330` | 2026-06-05 | Anbeeld | Fix KVarN prompt-cache checkpoint restore | 16 | src/(11), tools/server(3), tests/(2) | cache,prompt |
| `6914c6ec9` | 2026-06-05 | Anbeeld | fix(ci): clean up KVarN cache warnings | 4 | src/(4) | cache |
| `ee1f89c08` | 2026-06-05 | Anbeeld | fix(dflash): stabilize long-prompt KVarN serving | 10 | tests/(4), tools/server(3), common/(2) | prompt |
| `ef404570b` | 2026-06-05 | Anbeeld | fix(bench): accept KVarN cache types | 3 | other(1), tests/(1), tools/other(1) | cache |
| `645cfb79a` | 2026-06-05 | Anbeeld | fix(kvarn): guard prompt-cache rollback by memory capability | 18 | src/(14), common/(1), other(1) | cache,prompt,memory |
| `7119926d1` | 2026-06-05 | Anbeeld | fix(kvarn): allow full prompt-cache sequence clears | 4 | src/(3), tests/(1) | cache,prompt |
| `7c78c6f89` | 2026-06-05 | Anbeeld | perf(cuda): precompute KVarN materialization group counts | 1 | ggml/src/ggml-cuda(1) | perf,cuda |
| `02a4b573e` | 2026-06-05 | Anbeeld | Fix KVarN KV policy and DFlash adaptive probing | 10 | tests/(4), tools/server(3), src/(2) | kv |
| `47ab6bf62` | 2026-06-04 | Anbeeld | Use cache-type pseudo names for KVarN | 3 | common/(2), tests/(1) | cache |
| `17d0cb084` | 2026-06-04 | Anbeeld | Add baseline KVarN KV cache support | 28 | src/(10), tests/(4), common/(3) | cache,kv |
| `365e58516` | 2026-06-04 | Anbeeld | fix: repair fused turbo MMA cache paths | 40 | ggml/src/ggml-cuda(36), tests/(3), src/(1) | mma,cache,fused |
| `02dc2ee89` | 2026-06-04 | Anbeeld | ci: enable all-quant flash attention for HIP | 1 | .github(1) | flash,quant |
| `178404e12` | 2026-06-02 | Anbeeld | Revert "Add MXFP6 type and move TurboQuant KV IDs" | 21 | ggml/src/other(6), ggml/src/ggml-cpu(5), tests/(4) | kv |
| `a47cd8e0a` | 2026-06-02 | Anbeeld | Add MXFP6 type and move TurboQuant KV IDs | 21 | ggml/src/other(6), ggml/src/ggml-cpu(5), tests/(4) | kv |
| `c0f230856` | 2026-06-02 | Anbeeld | server: defer DFlash accept KV maintenance | 2 | tests/(1), tools/server(1) | kv |
| `6478051e2` | 2026-06-01 | Anbeeld | Enable fused GDN probing on ROCm | 2 | src/(1), tests/(1) | fused |
| `202f464c1` | 2026-06-01 | Anbeeld | Make DFlash multi-slot shared drafting opt-in | 4 | docs(2), tests/(1), tools/server(1) | opt |
| `0ef8c42f4` | 2026-06-01 | Anbeeld | ggml-cpu: reject unsupported BF16 scale | 3 | ggml/src/ggml-cpu(1), src/models(1), tests/(1) | bf16,cpu |
| `9f268fb03` | 2026-05-31 | Anbeeld | fix: reject unsupported CPU flash attention types | 4 | ggml/src/ggml-cpu(2), tests/(2) | flash,cpu |
| `0661d9eda` | 2026-05-31 | Anbeeld | ggml-cpu: fix fatal warning cases | 1 | ggml/src/ggml-cpu(1) | cpu |
| `2a6fd1475` | 2026-05-31 | Anbeeld | Fix CUDA GDN recurrent snapshots | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda |
| `c37f8056a` | 2026-05-26 | Anbeeld | Defer DFlash drafter KV updates while adaptive off | 4 | common/(2), tests/(1), tools/server(1) | kv |
| `2446c0ed4` | 2026-05-26 | Anbeeld | Add q6_0 KV cache support | 45 | ggml/src/ggml-cuda(32), ggml/src/other(4), ggml/src/ggml-cpu(4) | cache,kv |
| `d6b0dba1f` | 2026-05-26 | Anbeeld | cuda: gate fused Turbo MMA to decode batches | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda,mma,decode,fused |
| `413b33253` | 2026-05-26 | Anbeeld | metal: split turbo4 set_rows kernel | 2 | ggml/src/ggml-metal(1), tests/(1) | kernel |
| `a3508995e` | 2026-05-25 | Anbeeld | dflash: disable shared GPU kv cache for multi-slot drafter | 3 | src/(2), tests/(1) | cache,kv |
| `55ea235f4` | 2026-05-24 | Anbeeld | spec: remove stale dflash arg aliases | 4 | common/(2), tests/(1), tools/server(1) | spec |
| `9b2b1d137` | 2026-05-24 | Anbeeld | spec: align dflash defaults with upstream args | 4 | common/(2), tests/(1), tools/server(1) | spec |
| `2863d5b97` | 2026-05-24 | Anbeeld | dflash: reapply shared prefill capture state | 2 | common/(1), tests/(1) | prefill |
| `37175d22a` | 2026-05-24 | Anbeeld | dflash: avoid mixed non-spec np target batches | 3 | src/(1), tests/(1), tools/server(1) | spec |
| `76e3f43de` | 2026-05-24 | Anbeeld | cpu: support f16 out_prod fallback | 5 | ggml/src/ggml-cpu(3), tests/(2) | cpu |
| `f47747466` | 2026-05-24 | Anbeeld | cuda: enable dflash peer d2d copies | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda |
| `d237e4e47` | 2026-05-24 | Anbeeld | cmake: reject obsolete CUDA arch option | 2 | cmake(1), tests/(1) | cuda |
| `111b8eb51` | 2026-05-24 | Anbeeld | cuda: keep HIP TCQ attention on native vector path | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda |
| `ac792986d` | 2026-05-24 | Anbeeld | ggml: preserve fused GDN 4D state fast path | 10 | ggml/src/other(2), src/models(2), tests/(2) | fused |
| `78cf9fc4f` | 2026-05-24 | Anbeeld | models: restore Qwen per-layer KV heads | 3 | src/models(2), tests/(1) | kv |
| `0881616bd` | 2026-05-24 | Anbeeld | graph: gate pre-norm hidden output | 2 | src/(1), tests/(1) | graph |
| `5678fc3b8` | 2026-05-24 | Anbeeld | cuda: restore D512 flash attention selector | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda,flash |
| `4d2718978` | 2026-05-24 | Anbeeld | common: keep DFlash and CopySpec opt-in | 2 | common/(1), tests/(1) | opt |
| `97226547d` | 2026-05-24 | Anbeeld | common: avoid DFlash prompt raw-logit output | 2 | common/(1), tests/(1) | prompt |
| `ff6aae0ce` | 2026-05-23 | Anbeeld | mtmd: cap Gemma3 non-causal image decode | 2 | tests/(1), tools/other(1) | decode |
| `91357ddcc` | 2026-05-22 | Developer | fix(cuda): increase argmax topk kernel limit from 32 to 64 | 1 | ggml/src/ggml-cuda(1) | kernel,cuda,topk |
| `a1080e985` | 2026-05-21 | Anbeeld | dflash: clamp accepted-prefix full-KV commits to drafter window | 2 | common/(1), tests/(1) | kv |
| `71780eabe` | 2026-05-21 | Anbeeld | gemma4-iswa: port the upstream graph layout onto Bee hooks | 1 | src/models(1) | graph |
| `de68cd5a5` | 2026-05-21 | Anbeeld | cuda: restore upstream D=512 flash-attn selection | 1 | ggml/src/ggml-cuda(1) | cuda,flash,attn |
| `0b29e71a1` | 2026-05-21 | Anbeeld | server: hide routine decode timing behind debug logging | 1 | tools/server(1) | decode |
| `685f9bb61` | 2026-05-20 | Anbeeld | Restore legacy perplexity logits cache magic | 2 | tests/(1), tools/other(1) | cache |
| `c5eefecf8` | 2026-05-19 | Anbeeld | Bound Gemma4 image decode by decoder ubatch | 10 | tools/other(7), tests/(2), tools/server(1) | ubatch,decode |
| `72723812d` | 2026-05-19 | Anbeeld | Version perplexity logits cache format | 2 | tests/(1), tools/other(1) | cache |
| `90bf7b587` | 2026-05-19 | Anbeeld | Fail closed on unsafe DFlash prefill state | 4 | common/(2), tests/(1), tools/server(1) | prefill |
| `eac216020` | 2026-05-18 | Anbeeld | server: fix recurrent prompt cache and scheduling | 1 | tools/server(1) | cache,prompt |
| `fe54d13a0` | 2026-05-18 | Anbeeld | dflash: order backup and KV copies on CUDA streams | 5 | src/(2), other(1), tests/(1) | cuda,kv |
| `a15ec867e` | 2026-05-18 | Anbeeld | dflash: cache recurrent CUDA copy plans | 7 | src/(7) | cuda,cache |
| `853344ea8` | 2026-05-18 | Anbeeld | dflash: expose CUDA stream ordering helpers | 2 | ggml/src/ggml-cuda(2) | cuda |
| `00768f849` | 2026-05-17 | Anbeeld | cuda: reuse dflash stream wait events | 2 | ggml/src/ggml-cuda(1), tests/(1) | cuda |
| `75cb22750` | 2026-05-17 | Anbeeld | llama: fix dflash graph reuse state | 3 | src/(2), tests/(1) | graph |
| `eec368efd` | 2026-05-17 | Anbeeld | cuda: support d512 quant kv flash attention | 73 | ggml/src/ggml-cuda(72), tests/(1) | cuda,flash,kv,quant |
| `d9bbdaa7f` | 2026-05-17 | Anbeeld | tests: cover multi-slot dflash prefill plumbing | 1 | tests/(1) | prefill |
| `a85c675e1` | 2026-05-17 | Anbeeld | server: scope dflash prefill scheduling per slot | 3 | common/(2), tools/server(1) | prefill |
| `350eb3f74` | 2026-05-17 | Anbeeld | dflash: track prefill capture per slot | 7 | src/(4), src/models(3) | prefill |
| `f109314bc` | 2026-05-17 | Anbeeld | dflash: fix prefill staging to use planned suffix span, fail-closed tree update under GPU-only capture, debug-gate indexed mismatch, remove unused local | 5 | common/(2), src/(2), tests/(1) | prefill |
| `eb6696f99` | 2026-05-16 | Anbeeld | dflash: add source-aware CPU-ring validity tests, callback suppression tests, and plumbing assertions | 1 | tests/(1) | cpu |
| `083852f53` | 2026-05-16 | Anbeeld | dflash: suppress eval callback for no-intersection prefill ubatches, debug-gate per-ubatch logs | 2 | src/(2) | ubatch,prefill |
| `f99bc7bed` | 2026-05-16 | Anbeeld | dflash: make ring_write CPU-ring validity source-aware, fix flush_prefill force_cpu_ring, add verify_gpu source | 2 | common/(2) | cpu |
| `fdfa1ebf6` | 2026-05-16 | Anbeeld | dflash: fix capture_layers shadow bug, refuse large prefill CPU fallback, add GGML_DFLASH_DEBUG env | 2 | common/(2) | prefill,cpu |
| `424273cb4` | 2026-05-16 | Anbeeld | dflash: fix graph capture blocks to use dflash_capture_n_tokens/n_seqs | 3 | src/models(3) | graph |
| `7e87e4b59` | 2026-05-16 | Anbeeld | dflash: add plumbing tests for Gemma4 prefill, partial capture guard, GPU multi-slot, mask validation, cpu_ring_valid | 1 | tests/(1) | prefill |
| `b70b410a0` | 2026-05-16 | Anbeeld | dflash: disable DFlash capture on prefill flush mismatch instead of just logging | 1 | tools/server(1) | prefill |
| `a8c02ff0f` | 2026-05-16 | Anbeeld | dflash: add missing Gemma4 prefill_gpu staging graph copies | 1 | src/models(1) | graph |
| `49ce515c6` | 2026-05-16 | Anbeeld | dflash: harden prefill - graph reuse keys on src/dst offsets, per-view capture plan, cparams cleanup | 2 | src/(2) | graph,prefill |
| `2517050a8` | 2026-05-16 | Anbeeld | dflash: update plumbing test for prefill capture plan assertions | 1 | tests/(1) | prefill |
| `2749a74cd` | 2026-05-16 | Anbeeld | dflash: add prefill capture plan - accumulated window staging across internal ubatches | 7 | src/(4), src/models(2), other(1) | prefill |
| `b4b09737a` | 2026-05-16 | Anbeeld | dflash: fix prefill GPU flush - source-first control flow and source-aware ring_write | 2 | common/(1), tests/(1) | prefill |
| `fb0164b0a` | 2026-05-16 | Anbeeld | dflash: fix and extend plumbing test for prefill staging assertions | 1 | tests/(1) | prefill |
| `0d350833a` | 2026-05-16 | Anbeeld | dflash: persist pre-decode flush decision so suffix actually reaches the ring | 3 | common/(2), tools/server(1) | decode |
| `b4a731d93` | 2026-05-16 | Anbeeld | dflash: fix prefill GPU staging flush - use prefill_gpu_n_tokens when callback is null | 4 | src/(2), common/(1), other(1) | prefill |
| `16b19ccd8` | 2026-05-16 | Anbeeld | dflash: add prefill GPU staging buffer and suffix span math | 10 | src/(4), common/(2), src/models(2) | prefill |
| `c5551ff2a` | 2026-05-16 | Anbeeld | dflash: add logical capture toggle, fix prefill capture disabling GPU buffers | 5 | src/(2), common/(1), other(1) | prefill |
| `6b71fa58e` | 2026-05-16 | Anbeeld | Skip useless DFlash hidden capture and ring maintenance during early prompt prefill | 5 | common/(2), src/(1), tests/(1) | prefill,prompt |
| `fbd6c0d13` | 2026-05-16 | Anbeeld | test(dflash-plumbing): add checks for runtime diagnostics, shape validation, env toggles, CUDA debug | 1 | tests/(1) | cuda |
| `489f74115` | 2026-05-16 | Anbeeld | feat(cuda-dflash): add GGML_DFLASH_CUDA_DEBUG env and diagnostic logs | 1 | ggml/src/ggml-cuda(1) | cuda |
| `16622ac36` | 2026-05-16 | Anbeeld | feat(dflash): add force-cpu-cross/verbose-contract env toggles and expand GPU ring policy log | 1 | common/(1) | cpu |
| `d948b9f6f` | 2026-05-20 | Developer | fix: include --spec-draft-ctx-size in DFlash arg_passed check | 1 | common/(1) | spec |
| `4db14be0a` | 2026-05-17 | Anbeeld | Fix streaming perplexity logits memory | 1 | tools/other(1) | memory |
| `2b9aa77aa` | 2026-05-17 | Anbeeld | Fix CUDA driver link propagation | 1 | ggml/src/ggml-cuda(1) | cuda |
| `e50d29eed` | 2026-05-17 | LPFchan | fix: DFlash GPU ring hang with turbo4 - sync ggml backend after decode | 1 | src/(1) | decode |
| `75ae2a6af` | 2026-05-15 | Anbeeld | Fix GPU ring crash in DFlash KV update at long context prefill (#16) | 2 | src/(2) | kv,prefill |
| `52462e134` | 2026-05-14 | Anbeeld | Reduce peak memory in perplexity tool | 1 | tools/other(1) | memory |
| `da67e7411` | 2026-05-13 | Anbeeld | Harden DFlash recurrent replay on split CUDA | 4 | src/(2), ggml/src/ggml-cuda(1), tests/(1) | cuda |
| `fca6b9350` | 2026-05-12 | Anbeeld | Harden DFlash CUDA multi-GPU fallbacks | 6 | src/(2), common/(1), ggml/src/ggml-cuda(1) | cuda |
| `fdfd79e63` | 2026-05-11 | Anbeeld | Harden DFlash multi-GPU graph fallbacks | 3 | common/(1), ggml/src/ggml-cuda(1), tests/(1) | graph |
| `3c68b44ad` | 2026-05-03 | spiritbuun | server: fix prompt caching for hybrid models (Qwen3.6-35B-A3B) | 1 | tools/server(1) | prompt |
| `a12fd999c` | 2026-05-03 | Gastón Parravicini | ggml-cuda: extend zero-dim guard to cover ne00 and ldc | 1 | ggml/src/ggml-cuda(1) | cuda |
| `3416ff13f` | 2026-05-02 | Gastón Parravicini | ggml-cuda: fix cublasSgemm crash with zero-dim matrices in speculative decoding | 1 | ggml/src/ggml-cuda(1) | cuda |
| `239296c43` | 2026-05-01 | Steffen Röcker | cuda: extend Ada MMQ tile cap to small IQ quants | 1 | ggml/src/ggml-cuda(1) | cuda |
| `ab5a019bc` | 2026-05-01 | Steffen Röcker | cuda: tune Ada MMQ tile cap for small quants | 1 | ggml/src/ggml-cuda(1) | cuda |
| `aecbbd5da` | 2026-05-01 | spiritbuun | fix: CI build errors on Android and Windows CUDA 12.4 | 3 | ggml/src/ggml-cuda(2), ggml/src/other(1) | cuda |
| `905483277` | 2026-05-01 | spiritbuun | add --no-fused-gdn flag for debugging fused kernel issues | 5 | common/(3), other(1), src/(1) | kernel,fused |
| `115995e41` | 2026-05-01 | spiritbuun | fix: disable fused GDN kernels on non-CUDA backends (fixes #36) | 1 | src/(1) | cuda,fused |
| `a9cd1ec3e` | 2026-04-30 | spiritbuun | cpu: add SSM_CONV_TREE and GATED_DELTA_NET_TREE fallback implementations | 4 | ggml/src/ggml-cpu(3), src/(1) | cpu |
| `2e239fbcb` | 2026-04-30 | spiritbuun | perf: port turbo3_tcq optimizations to turbo2_tcq encoder | 2 | ggml/src/ggml-cuda(2) | perf |
| `e275191e8` | 2026-04-29 | spiritbuun | perf: use exp2 SFU fast path for GDN gate decay | 1 | ggml/src/ggml-cuda(1) | perf |
| `12a648efc` | 2026-04-29 | Steffen Röcker | cuda: streamline tcq final state selection | 1 | ggml/src/ggml-cuda(1) | cuda |
| `018092c45` | 2026-04-28 | Steffen Röcker | cuda: optimize turbo3 tcq set_rows | 2 | ggml/src/ggml-cuda(2) | optimize,cuda |
| `7bdc544b0` | 2026-04-27 | spiritbuun | SD-084v2: defer recurrent state backup cells during prefill | 6 | src/(4), other(1), tools/server(1) | prefill |
| `556688392` | 2026-04-26 | spiritbuun | guard cudaFuncSetAttribute in turbo MMA for HIP | 1 | ggml/src/ggml-cuda(1) | mma |
| `30759dfa0` | 2026-04-26 | spiritbuun | fix DFlash slot selection: prefer spec-capable slots over LRU | 1 | tools/server(1) | spec |
| `9a7ff084d` | 2026-04-25 | spiritbuun | fix multi-GPU CUDA errors and mmproj+drafter conflict | 3 | ggml/src/ggml-cuda(2), tools/server(1) | cuda |
| `a6fcec7c7` | 2026-04-25 | spiritbuun | fused MMA turbo4 flash attention: read raw turbo bytes directly in MMA kernel | 36 | ggml/src/ggml-cuda(36) | kernel,mma,flash,fused |
| `2a491077d` | 2026-04-25 | spiritbuun | draft model warmup + prefer spec-capable slots in LRU selection | 2 | common/(1), tools/server(1) | spec |
| `ff0444e46` | 2026-04-25 | spiritbuun | B2.4+B2.5: batched DFlash draft - single drafter decode for all concurrent slots | 4 | common/(2), src/models(1), tools/server(1) | decode |
| `4f995ac92` | 2026-04-25 | spiritbuun | B2.3c: dynamic drafter graph narrowing for active slot count | 1 | tools/server(1) | graph |
| `b00c58e20` | 2026-04-24 | spiritbuun | dflash: runtime knob for drafter graph ctx_len (B2.2) | 6 | src/(3), other(1), src/models(1) | graph |
| `50eefd187` | 2026-04-24 | spiritbuun | dflash: widen drafter graph to MAX_SLOTS x block_size (B2.1) | 2 | src/models(1), tools/server(1) | graph |
| `2cc97a81c` | 2026-04-24 | spiritbuun | server: allow assistant-response prefill with enable_thinking | 1 | tools/server(1) | prefill |
| `8f75fb657` | 2026-04-24 | spiritbuun | dflash-server: move tape_recording(false) out of sub-batch loop | 1 | tools/server(1) | batch |
| `ee3a2ff49` | 2026-04-24 | spiritbuun | dflash-server: fix prefill hidden-state starvation (+12pp accept) | 1 | tools/server(1) | prefill |
| `53d98bd1b` | 2026-04-24 | spiritbuun | dflash: make -ub cap discoverable + decouple drafter graph size | 2 | common/(1), tools/server(1) | graph |
| `6afafd95d` | 2026-04-24 | spiritbuun | fix: pass seq_id through DFlash rollback/tape_replay (M-RoPE multi-slot) | 5 | src/(2), examples/(1), other(1) | rope |
| `dcccd9e68` | 2026-04-24 | flamme-demon | turbo-quant/turbo-sink: include vendor shim on HIP builds | 3 | ggml/src/ggml-cuda(3) | quant |
| `fa7b2d6f3` | 2026-04-24 | flamme-demon | ggml-cuda: widen warp shuffle mask to 64-bit | 10 | ggml/src/ggml-cuda(10) | cuda |
| `90bd5ea81` | 2026-04-24 | flamme-demon | hip.h: use native ROCm 7.2 shfl templates + add memcpy symbol aliases | 1 | ggml/src/ggml-cuda(1) | memcpy |
| `22464d084` | 2026-04-23 | spiritbuun | dflash: auto-apply memory-safe defaults for -b / -ub / -cd | 1 | common/(1) | memory |
| `95c557813` | 2026-04-23 | Chris | CUDA: fix turbo cache types producing garbage on multi-GPU setups (#18) | 2 | ggml/src/ggml-cuda(2) | cuda,cache |
| `8094fc416` | 2026-04-21 | spiritbuun | SD-071: ubatch fix + best-first heap + cleanup | 6 | src/(2), common/(1), examples/(1) | ubatch |
| `729ded76f` | 2026-04-20 | spiritbuun | SD-071: DDTree KV cache fix - eliminate re-decode after tree verification | 14 | src/(4), common/(3), src/models(2) | cache,kv,decode |
| `fa98f424c` | 2026-04-20 | spiritbuun | fix: skip tree mode when batch size exceeds parent_ids tensor | 2 | src/models(2) | batch |
| `f38d36dbf` | 2026-04-20 | spiritbuun | SD-069: async tape replay overlaps GDN kernel with draft generation (+2-3%) | 5 | src/(2), examples/(1), other(1) | kernel |
| `446320565` | 2026-04-20 | spiritbuun | GPU argmax for DFlash drafter: eliminate 15.9MB transfer + CPU scan | 7 | src/(4), common/(1), other(1) | transfer,cpu |
| `2a6f9480d` | 2026-04-18 | spiritbuun | Fix drafter OOM: cap batch size to 64 (drafter only uses block_size=16) | 1 | src/models(1) | batch |
| `edecf9699` | 2026-04-18 | spiritbuun | Fix drafter OOM: cap batch size to 64 (drafter only uses block_size=16) | 1 | examples/(1) | batch |
| `d16a50d3b` | 2026-04-18 | spiritbuun | SD-060: DDTree MVP results - tree acceptance up but re-eval kills speed | 1 | examples/(1) | speed |
| `297a1cd83` | 2026-04-15 | spiritbuun | optimize copy_cell: batch async D2D copies with single sync point | 2 | src/(2) | optimize,batch |
| `d20b090d7` | 2026-04-15 | spiritbuun | Save/restore recurrent state for correct hybrid spec decode | 2 | examples/(1), src/(1) | spec,decode |
| `74ea00da0` | 2026-04-15 | spiritbuun | Simplify recurrent state handling for speculative decode | 1 | src/(1) | decode |
| `193ff0acd` | 2026-04-15 | spiritbuun | Fix speculative decode rollback for hybrid models | 2 | src/(2) | decode |
| `1cd508d1d` | 2026-04-15 | spiritbuun | Relax M-RoPE position validation for speculative decode re-eval | 1 | src/(1) | decode,rope |
| `800e3cd8d` | 2026-04-15 | spiritbuun | C.0: Allow model-free spec types in speculative-simple | 1 | examples/(1) | spec |
| `63aa3591e` | 2026-04-10 | Aman Gupta | CUDA: fuse muls (#21665) | 3 | ggml/src/ggml-cuda(3) | cuda |
| `6de21bf5e` | 2026-04-09 | Aman Gupta | CUDA: also store `node->src->data` ptrs for equality check (#21635) | 2 | ggml/src/ggml-cuda(2) | cuda |
| `914e5c254` | 2026-04-07 | Gaurav Garg | [CUDA ] Write an optimized flash_attn_stream_k_fixup kernel (#21159) | 1 | ggml/src/ggml-cuda(1) | kernel,cuda |
| `9f82ccf0e` | 2026-04-09 | spiritbuun | feat: turbo TCQ prompt cache safety fingerprint | 1 | src/(1) | cache,prompt |
| `2195068f8` | 2026-04-09 | spiritbuun | perf: shfl-based inv-FWHT butterfly + half2 stores for turbo K dequant | 1 | ggml/src/ggml-cuda(1) | perf |
| `59b7ea29b` | 2026-04-09 | spiritbuun | fix: size turbo dequant buffers from cache root to prevent realloc churn | 1 | ggml/src/ggml-cuda(1) | cache |
| `c51c47fef` | 2026-04-08 | Erik Scholz | kv-cache : extend cache quantization checks (#21586) | 1 | src/(1) | cache,kv |
| `69618c45b` | 2026-04-07 | Antoine Viallon | ggml-cuda : fix CDNA2 compute capability constant for gfx90a (MI210) (#21519) | 1 | ggml/src/ggml-cuda(1) | cuda |
| `1eaea4494` | 2026-04-08 | Aman Gupta | CUDA: make cuda graphs props check faster (#21472) | 2 | ggml/src/ggml-cuda(2) | faster,cuda |
| `466dc2cd5` | 2026-04-08 | Aman Gupta | CUDA: check for buffer overlap before fusing (#21566) | 1 | ggml/src/ggml-cuda(1) | cuda |
| `c3e4ef410` | 2026-04-07 | iacopPBK | ggml-cuda: ds_read_b128 for q4_0 and q4_1 mmq kernels (#21168) | 1 | ggml/src/ggml-cuda(1) | cuda |
| `f68275245` | 2026-04-07 | Georgi Gerganov | kv-cache : support attention rotation for heterogeneous iSWA (#21513) | 4 | src/(4) | cache,kv |
| `6bdafb23b` | 2026-04-08 | spiritbuun | fix: turbo4 K/V decode dequant for mixed K/V configurations (Bug B) | 1 | ggml/src/ggml-cuda(1) | decode |
| `f9a40e588` | 2026-04-06 | spiritbuun | feat: calibrated 2-bit adaptive decode-time V alpha from fine-grained KLD sweeps | 3 | other(2), ggml/src/ggml-cuda(1) | decode |
| `3c4c44e5c` | 2026-04-05 | spiritbuun | simplify: consolidate codebook load loop, remove hardcoded sizes | 1 | ggml/src/ggml-cuda(1) | load |
| `692cffde1` | 2026-04-05 | spiritbuun | experiment: S3 TCQ codebook from __constant__ to __shared__ memory | 2 | ggml/src/ggml-cuda(2) | memory |
| `9d2748d78` | 2026-04-05 | spiritbuun | feat: context-adaptive decode-time V alpha scaling for TCQ | 3 | ggml/src/ggml-cuda(2), other(1) | decode |
| `dc5189961` | 2026-04-05 | spiritbuun | fix: BF16 precision for Gemma 4 scale ops (ported from upstream #21451) | 6 | ggml/src/ggml-cuda(4), other(1), src/models(1) | bf16 |
| `29dc360f1` | 2026-04-04 | spiritbuun | perf: double-buffered Viterbi cost + global backtrace for TCQ encode | 2 | ggml/src/ggml-cuda(2) | perf |
| `f120f3d9d` | 2026-04-03 | spiritbuun | feat: decode-time V alpha, golden codebooks, Q² calibration + full KLD campaign | 11 | other(4), ggml/src/ggml-cuda(4), scripts/(3) | decode |
| `6e2f5a71d` | 2026-04-01 | spiritbuun | fix: TCQ vec decode dispatch + honest alpha defaults | 6 | ggml/src/ggml-cuda(4), other(2) | decode |
| `177fcb3ae` | 2026-04-01 | spiritbuun | docs: mark experiment #73 done - 12.6% prefill speedup | 1 | other(1) | prefill |
| `20be16122` | 2026-04-01 | spiritbuun | docs: experiment #72 chunked cuBLAS GEMM prefill - rejected (1-5% slower) | 2 | other(2) | cublas,prefill,gemm |
| `fdd18371e` | 2026-03-31 | spiritbuun | feat: TCQ temperature scaling alpha=1.20 - 5-14% PPL improvement | 3 | other(2), ggml/src/ggml-cuda(1) | improvement |
| `e588de031` | 2026-03-31 | spiritbuun | fix: load TURBO_TCQ_ALPHA in 2-bit TCQ branch too | 1 | ggml/src/ggml-cuda(1) | load |
| `14bf515f3` | 2026-03-31 | spiritbuun | fix: move d_tcq_norm_alpha declaration before 3-bit TCQ kernel | 1 | ggml/src/ggml-cuda(1) | kernel |
| `c440e673b` | 2026-03-31 | spiritbuun | feat: runtime codebook loading, V decode fix, codebook binaries, benchmarks | 48 | other(27), scripts/(15), ggml/src/ggml-cuda(6) | decode |
| `b063df99e` | 2026-03-28 | spiritbuun | simplify: remove redundant shared memory codebook, clean up TCQ kernel | 2 | ggml/src/ggml-cuda(2) | kernel,memory |
| `5047b3e9a` | 2026-03-28 | spiritbuun | fix: Windows MSVC compile error - turbo quant linkage mismatch (C2375) | 1 | ggml/src/ggml-cpu(1) | quant |
| `660221c19` | 2026-03-28 | spiritbuun | feat: turbo4 inverse-FWHT prefill dequant - 2.67x prefill speedup | 3 | other(2), ggml/src/ggml-cuda(1) | prefill |
| `06a89ddb3` | 2026-03-28 | spiritbuun | fix: turbo4 MMA prefill bypass + FA auto-enable | 7 | ggml/src/ggml-cuda(4), other(1), scripts/(1) | mma,fa,prefill |
| `02f4948b1` | 2026-03-28 | Will Hampson | docs: add multi-GPU long-context decode benchmark report | 1 | docs(1) | decode |
| `8e6ced0c3` | 2026-03-28 | spiritbuun | fix: turbo KV cache with --fit on / partial GPU offload (OOM regression) | 1 | src/(1) | cache,kv,offload |
| `d59af969c` | 2026-03-28 | spiritbuun | feat: InnerQ per-channel equalization for turbo3 KV cache | 4 | ggml/src/ggml-cuda(4) | cache,kv |
| `6f8f9234e` | 2026-03-27 | spiritbuun | fix: turbo dequant handles multi-stream KV (n_seq > 1) | 3 | other(2), ggml/src/ggml-cuda(1) | kv |
| `309b77e58` | 2026-03-27 | spiritbuun | feat: add GGML_TYPE_TURBO2_0 - 2-bit TurboQuant KV cache (2.5 bpv, 6.4x compression) | 27 | ggml/src/ggml-cuda(13), ggml/src/other(4), src/(3) | cache,kv |
| `ef588f8bf` | 2026-03-27 | spiritbuun | fix: error on turbo KV cache without Flash Attention | 1 | src/(1) | flash,cache,kv |
| `c99c23018` | 2026-03-27 | spiritbuun | fix: error on turbo KV cache with partial GPU offload (ngl < n_layer) | 3 | ggml/src/ggml-cpu(2), src/(1) | cache,kv,offload |
| `6cdd9db87` | 2026-03-27 | spiritbuun | fix: multi-GPU q_rot_buf + KV cache tensor count margin | 2 | ggml/src/ggml-cuda(1), src/(1) | cache,kv |
| `1010625c5` | 2026-03-27 | spiritbuun | perf: enable turbo4 prefill MMA - pp4096 588->1113 tok/s (1.9x) | 3 | other(2), ggml/src/ggml-cuda(1) | perf,mma,prefill |
| `271deea8c` | 2026-03-27 | spiritbuun | docs: benchmark results for turbo4 fix, sparse V, multi-model validation | 2 | other(2) | sparse |
| `78d6bb5a0` | 2026-03-27 | spiritbuun | fix: turbo4 K Q pre-rotation, iSWA V un-rotation, KV cache OOM + sparse V dequant | 4 | ggml/src/ggml-cuda(2), src/(2) | cache,kv,sparse |
| `7c6250688` | 2026-03-26 | spiritbuun | perf: dequant turbo3 KV to fp16 for decode - eliminates MoE context scaling | 2 | other(1), ggml/src/ggml-cuda(1) | perf,moe,kv,decode |
| `cffa06f63` | 2026-03-26 | spiritbuun | perf: move Q FWHT rotation out of vec kernel - decode +6.5% on MoE | 2 | ggml/src/ggml-cuda(2) | perf,kernel,moe,decode |
| `f68ad76b6` | 2026-03-26 | spiritbuun | feat: inline FWHT Q pre-rotation in FA kernels | 7 | ggml/src/ggml-cuda(3), other(2), src/(1) | fa |
| `b18c95f2b` | 2026-03-26 | spiritbuun | docs: add TurboQuant CUDA overview and benchmarks to README | 1 | other(1) | cuda |
| `0cd3b51e4` | 2026-03-26 | spiritbuun | docs: comprehensive LA + prefill MMA benchmark results | 1 | other(1) | mma,prefill |
| `4914d6a0d` | 2026-03-26 | spiritbuun | docs: update experiment #16 with turbo4 prefill findings | 1 | other(1) | prefill |
| `b9f776136` | 2026-03-26 | spiritbuun | fix: turbo4 prefill dequant block indexing + disable turbo4 MMA prefill | 2 | other(1), ggml/src/ggml-cuda(1) | mma,prefill |
| `375555536` | 2026-03-26 | spiritbuun | perf: turbo prefill dequant+MMA - 1.78x speedup (98.8% of q8_0) | 3 | ggml/src/ggml-cuda(3) | perf,mma,prefill |
| `6b821a94d` | 2026-03-26 | spiritbuun | perf: turbo4 norm correction - zero-cost quality improvement | 1 | ggml/src/ggml-cuda(1) | perf,improvement |
| `e6a78d542` | 2026-03-26 | spiritbuun | perf: turbo4 batch unpack + V dequant optimization + V_DOT2 half2 path | 1 | ggml/src/ggml-cuda(1) | perf,batch |
| `721880c00` | 2026-03-26 | spiritbuun | feat: FWHT rotation + optimized dequant for turbo3/turbo4 CUDA | 7 | ggml/src/ggml-cuda(7) | cuda |
| `12f1bc0bd` | 2026-03-26 | spiritbuun | feat: add CUDA support for turbo3 and turbo4 KV cache quantization | 15 | ggml/src/ggml-cuda(14), other(1) | cuda,cache,kv |
| `dfc109798` | 2026-03-26 | TheTom | fix: add turbo3/turbo4 cache types to llama-bench arg parser | 1 | tools/other(1) | cache |
| `aa6a3a180` | 2026-03-25 | TheTom | perf: float norm broadcast in vec dequant - decode +2-3% over fp16 LUT | 1 | ggml/src/ggml-metal(1) | perf,decode |
| `654647aac` | 2026-03-25 | TheTom | perf: fp16 centroid LUT - decode +6-14% at long context (#33) | 1 | ggml/src/ggml-metal(1) | perf,decode |
| `9cd043108` | 2026-03-25 | TheTom | ci: quality+speed gate script - PPL + context scaling check before push | 1 | scripts/(1) | speed |
| `ccbac3fab` | 2026-03-25 | TheTom | perf: optimized turbo3 dequant - eliminates context scaling regression | 1 | ggml/src/ggml-metal(1) | perf |
| `70b2376ca` | 2026-03-25 | TheTom | fix: address Codex review on layer-adaptive - thread safety + underflow guard | 1 | src/(1) | thread |
| `f26d47ab2` | 2026-03-25 | TheTom | feat: layer-adaptive KV cache - q8_0 quality with 80% turbo3 compression | 1 | src/(1) | cache,kv |
| `c84e12419` | 2026-03-25 | TheTom | perf: block-32 + graph WHT - 2747 tok/s (1.02x q8_0!!!) | 2 | ggml/src/other(1), ggml/src/ggml-metal(1) | perf,graph |
| `676f929fe` | 2026-03-25 | TheTom | perf: graph-side WHT rotation - 2095 tok/s (0.78x q8_0, was 0.53x) | 2 | ggml/src/ggml-metal(1), src/(1) | perf,graph |
| `640e10eff` | 2026-03-25 | TheTom | perf: pre-packed half4 sign arrays - minor speedup (1411 -> 1424 tok/s) | 1 | ggml/src/ggml-metal(1) | perf |
| `e4e0bde36` | 2026-03-25 | TheTom | perf: vectorized half4 WHT butterfly - 31% speedup (1074 -> 1411 tok/s) | 1 | ggml/src/ggml-metal(1) | perf |
| `a0e8a65f2` | 2026-03-25 | TheTom | perf: fp16 WHT dequant + SIMD cooperative dequant - 45% speedup | 8 | docs(3), src/(3), ggml/src/ggml-metal(1) | perf |
| `a696962d1` | 2026-03-25 | TheTom | feat: block size 32 - 77.7 tok/s MoE (91% of q8_0), 17.0 Qwopus (97%) 🎉 | 3 | ggml/src/other(2), ggml/src/ggml-metal(1) | moe |
| `e89adfe6d` | 2026-03-25 | TheTom | docs: pre-rotate-queries implementation plan + speed ceiling 49 tok/s | 2 | docs(1), src/(1) | speed |
| `1921c4869` | 2026-03-25 | TheTom | docs: speed ceiling test - 49 tok/s without dequant rotation (4.6x gain) #23 | 1 | docs(1) | speed |
| `8d682bddc` | 2026-03-25 | TheTom | fix: inline turbo-wht.h - was causing CPU fallback, not Metal! #23 | 2 | ggml/src/ggml-metal(2) | cpu |
| `c7ccedefb` | 2026-03-25 | TheTom | docs: log threadgroup attempt - no speed improvement, rethinking #23 | 1 | docs(1) | speed,improvement |
| `4806cc866` | 2026-03-25 | TheTom | docs: log simd_broadcast attempt - no speed improvement #23 | 1 | docs(1) | speed,improvement |
| `8ab4031da` | 2026-03-25 | TheTom | docs: detailed speed investigation plan for TurboQuant Metal shader #23 | 1 | docs(1) | speed |
| `8e5e6632c` | 2026-03-25 | TheTom | fix: remove thread static from Metal dequantize, fix stale code #23 | 1 | ggml/src/ggml-metal(1) | thread |
| `bf0e223f9` | 2026-03-24 | TheTom | feat: Metal kernels for TurboQuant KV cache (turbo3, turbo4) #21 | 2 | ggml/src/ggml-metal(2) | cache,kv |
| `4ff5bf885` | 2026-03-24 | TheTom | WIP: add TurboQuant KV cache types (turbo3, turbo4) | 7 | ggml/src/other(5), common/(1), ggml/include(1) | cache,kv |

### (d) Остальные коммиты без перф-маркеров: top-120 по числу изменённых файлов (из 520 всего без маркеров)
| SHA | Дата | Автор | Subject | Изменено файлов | Ключевые директории | Перф-маркеры |
|---|---|---|---|---|---|---|
| `f7832345b` | 2026-07-10 | Anbeeld | Remove TurboQuant core support | 277 | ggml/src/ggml-cuda(234), src/(13), ggml/src/other(9) |  |
| `c9e746733` | 2026-07-10 | Anbeeld | Complete BeeLlama v0.4.0 upstream rebase | 217 | ggml/src/ggml-cuda(78), src/(31), tests/(22) |  |
| `1a285c4d1` | 2026-05-23 | Anbeeld | ggml: preserve TurboQuant and TCQ backends | 163 | ggml/src/ggml-cuda(134), ggml/src/ggml-metal(9), ggml/src/other(8) |  |
| `686e63aa3` | 2026-05-09 | Anbeeld | Restore BeeLlama local changes on cleaned history | 118 | .github(24), src/(20), other(19) |  |
| `80bb3c794` | 2026-05-23 | Anbeeld | core: preserve BeeLlama speculative decoding support | 56 | src/(30), common/(15), src/models(8) |  |
| `3d44648f2` | 2026-05-31 | Anbeeld | Simplify Bee release workflows | 51 | .github(47), other(4) |  |
| `bdee25a6a` | 2026-07-17 | Anbeeld | Harden v0.4.0 production readiness | 47 | src/(11), tests/(9), tools/server(9) |  |
| `dd668d9b8` | 2026-07-20 | Anbeeld | kvarn: add native attention across backends | 43 | ggml/src/ggml-cuda(8), src/(7), docs(6) |  |
| `dd3b8c880` | 2026-04-03 | spiritbuun | feat: Gemma 4 architecture support + head padding for non-128 head_dim | 36 | ggml/src/ggml-cuda(21), src/(13), src/models(2) |  |
| `491bb5a2a` | 2026-07-27 | Anbeeld | Enable sharded KVarN precision tails | 35 | other(10), src/(9), tests/(7) |  |
| `f2eb256a5` | 2026-04-06 | spiritbuun | clean: remove internal files for public release | 35 | other(23), scripts/(12) |  |
| `0ae59839d` | 2026-07-26 | Anbeeld | hip: enable capability-driven KVarN routes | 31 | ggml/src/ggml-cuda(15), docs(7), src/(4) |  |
| `40970cbde` | 2026-07-03 | Anbeeld | Rework KVarN head-wide rotation, SWA ring, and store staging | 30 | ggml/src/ggml-cuda(11), src/(7), ggml/src/ggml-cpu(3) |  |
| `5ea707215` | 2026-05-23 | Anbeeld | build: preserve BeeLlama workflow metadata | 28 | .github(23), other(2), cmake(2) |  |
| `0c5034c1b` | 2026-08-09 | Anbeeld | fix: make KVarN precision-tail fitting exact and bounded | 27 | common/(6), src/(6), tests/(6) |  |
| `e9eb7eec1` | 2026-08-28 | Anbeeld | ci: harden v0.4.4 release packaging | 25 | scripts/(8), .github(6), docs(4) |  |
| `c3ebc89de` | 2026-07-15 | Anbeeld | Implement faithful KVarN exact-tail storage | 25 | ggml/src/ggml-cuda(12), src/(9), ggml/src/ggml-vulkan(2) |  |
| `f1a9b5a19` | 2026-04-18 | spiritbuun | DFlash speculative decoding with GPU tape replay and optimizations | 22 | src/(11), common/(3), src/models(3) |  |
| `7dd274819` | 2026-03-29 | spiritbuun | feat: turbo2_tcq 2-bit trellis-coded quantization - PPL 6.05 at 2.25 bpv | 22 | ggml/src/ggml-cuda(6), ggml/src/other(4), scripts/(3) |  |
| `0fb10f6b4` | 2026-08-10 | Anbeeld | Updated documentation | 21 | docs(19), other(2) |  |
| `ff3954375` | 2026-07-14 | Anbeeld | Extend GGML backends for mixed-precision tail attention | 21 | ggml/src/ggml-cuda(15), ggml/src/ggml-cpu(2), ggml/CMakeLists.txt(1) |  |
| `5ecbe1ac1` | 2026-08-18 | Jian Chen | support DFlash2 | 20 | src/(7), common/(4), gguf-py(3) |  |
| `afaffd0e9` | 2026-05-23 | Anbeeld | server: preserve DFlash and mtmd integration | 20 | tools/server(9), tools/other(8), tests/(3) |  |
| `ffaa96f5a` | 2026-03-28 | spiritbuun | feat: turbo3_tcq - trellis-coded quantization (3.25 bpv, right-shift bitshift trellis) | 20 | ggml/src/ggml-cuda(6), ggml/src/other(4), src/(3) |  |
| `5ed70f660` | 2026-08-02 | Anbeeld | hip: make KVarN tail routing shape-safe | 17 | tests/(6), ggml/src/ggml-cuda(5), scripts/(2) |  |
| `eb1e51787` | 2026-06-19 | Anbeeld | Use zero-materialize KVarN native views | 17 | ggml/src/other(4), ggml/src/ggml-cuda(4), src/(3) |  |
| `bb87db3a6` | 2026-06-04 | Anbeeld | Extend KVarN to Qwen3.6 and Gemma4 | 16 | src/(15), tests/(1) |  |
| `bd755711e` | 2026-04-20 | spiritbuun | SD-070: tree-aware SSM kernels for single-pass DDTree verification | 16 | ggml/src/ggml-cuda(5), src/(4), src/models(2) |  |
| `9b9277f73` | 2026-07-17 | Anbeeld | Restore minimal BeeLlama release workflows | 15 | .github(14), tests/(1) |  |
| `df63a91df` | 2026-07-16 | Anbeeld | Remove legacy DFlash architecture support | 15 | other(7), docs(4), tests/(3) |  |
| `7d7a205f1` | 2026-07-16 | Anbeeld | Cover exact-tail lifecycle and route invariants adversarially | 14 | tests/(9), tools/server(4), cmake(1) |  |
| `ee1d1a308` | 2026-06-06 | Anbeeld | Fix flat DFlash recurrent rollback sizing | 14 | src/(7), common/(2), tests/(2) |  |
| `de37c756e` | 2026-06-05 | Anbeeld | Enable parallel KVarN streams | 14 | src/(7), tests/(2), ggml/include(1) |  |
| `5bd96147e` | 2026-05-23 | Anbeeld | tests: preserve BeeLlama safety-net coverage | 14 | tests/(12), examples/(1), tools/other(1) |  |
| `25901d670` | 2026-06-28 | Anbeeld | Fix KVarN stage-domain and native attention routing | 13 | ggml/src/ggml-cuda(6), src/(3), ggml/include(1) |  |
| `d590240f3` | 2026-06-20 | Anbeeld | Fix KVarN native capability gates | 13 | src/(6), ggml/src/ggml-cuda(3), other(1) |  |
| `69af3392b` | 2026-03-25 | TheTom | feat: add GGML_OP_TURBO_WHT - custom O(d log d) Walsh-Hadamard Transform | 13 | ggml/src/ggml-metal(7), ggml/src/ggml-cpu(3), ggml/include(1) |  |
| `185bc3de1` | 2026-05-16 | Anbeeld | Fix op table drift, harden GDN kernels, stream-safe DFlash replay | 12 | tests/(3), cmake(2), common/(2) |  |
| `068ebad6a` | 2026-04-20 | spiritbuun | SD-067/068: rejection sampling for temp>0 + CopySpec stacking on DFlash | 12 | common/(3), src/(3), examples/(1) |  |
| `a7d0f754b` | 2026-04-09 | Daniel Bevenius | requirements : update transformers to 5.5.1 (#21617) | 12 | other(6), examples/(2), tests/(2) |  |
| `005128c49` | 2026-08-28 | Anbeeld | Limit adaptive draft-max to DFlash1 | 11 | other(3), common/(3), docs(3) |  |
| `41d41b0ce` | 2026-07-19 | Anbeeld | Updated documentation for v0.4.0 | 11 | docs(7), other(4) |  |
| `c0882e10b` | 2026-07-11 | Anbeeld | Default DFlash depth to drafter metadata | 11 | common/(3), docs(3), other(2) |  |
| `d94f2f39d` | 2026-07-02 | Anbeeld | Fix KVarN rotated stage contract | 11 | ggml/src/ggml-cuda(4), tests/(3), src/(2) |  |
| `efe856397` | 2026-06-01 | Anbeeld | Fix reasoning loop guard enforcement | 11 | tools/server(5), common/(4), tests/(1) |  |
| `3ed96ba2e` | 2026-05-24 | Anbeeld | dflash: enable device-aware multi-gpu tape | 11 | src/(4), docs(2), common/(1) |  |
| `aa43e6b28` | 2026-05-21 | Anbeeld | DFlash: keep the drafter aligned with the live suffix | 11 | src/(6), ggml/src/ggml-cuda(2), common/(1) |  |
| `e22e4dac1` | 2026-05-20 | Anbeeld | dflash: make server depth control authoritative | 11 | src/(4), ggml/src/ggml-cuda(3), common/(2) |  |
| `1a765bdcb` | 2026-04-25 | spiritbuun | B3: cleanup - remove dev markers, dead code, fix bugs | 11 | src/(4), src/models(3), common/(2) |  |
| `cddecf698` | 2026-04-24 | spiritbuun | dflash: multi-slot server support (--dflash-max-slots) | 11 | src/(7), common/(2), other(1) |  |
| `9084f1bed` | 2026-07-28 | Anbeeld | build: standardize Windows build scripts | 10 | docs(4), scripts/(3), other(2) |  |
| `335f252ce` | 2026-06-28 | Anbeeld | KVarN: align live-stage domains with attention | 10 | ggml/src/ggml-cuda(4), src/(2), tests/(2) |  |
| `e211e4d92` | 2026-06-17 | Anbeeld | Bring KVarN to SWA layers: 100% coverage via sliding-window ring | 10 | src/(7), ggml/src/ggml-cpu(1), ggml/src/ggml-cuda(1) |  |
| `765d04753` | 2026-06-17 | Anbeeld | Make KVarN rotated attention the default | 10 | ggml/src/ggml-cuda(5), other(2), tests/(2) |  |
| `12a169b8f` | 2026-06-05 | Anbeeld | Add KVarN ROCm and Vulkan backend support | 10 | ggml/src/ggml-vulkan(4), ggml/src/ggml-cuda(3), tests/(2) |  |
| `552b701de` | 2026-06-03 | Anbeeld | ci: harden release packaging | 10 | other(5), .github(4), scripts/(1) |  |
| `10acb484b` | 2026-05-13 | Anbeeld | Fix profit controller baseline handling | 10 | tools/server(3), common/(2), docs(2) |  |
| `6aca44723` | 2026-04-18 | spiritbuun | SD-060: DDTree MVP - tree-structured speculative decoding for DFlash | 10 | common/(4), src/(4), examples/(1) |  |
| `9fbc13b34` | 2026-07-28 | Anbeeld | Harden v0.4.2 release readiness | 9 | src/(4), tests/(4), other(1) |  |
| `db7e2945f` | 2026-07-11 | Anbeeld | Fix KVarN staging for large logical batches | 9 | docs(4), src/(2), tests/(2) |  |
| `35c10e573` | 2026-06-04 | Anbeeld | fix: support DFlash tensor-split Meta placement | 9 | src/(2), other(1), common/(1) |  |
| `20e1ff940` | 2026-05-13 | Anbeeld | Stabilize profit controller depth policy | 9 | common/(2), docs(2), tests/(2) |  |
| `2d7a40c6b` | 2026-08-28 | Anbeeld | ci: sync v0.4.4 release workflow | 8 | tests/(4), scripts/(3), .github(1) |  |
| `1ce739a01` | 2026-08-27 | Anbeeld | Fix warning-as-error release builds | 8 | tests/(7), src/(1) |  |
| `6fcc81008` | 2026-08-02 | Anbeeld | ci: restore existing release workflows | 8 | tests/(4), scripts/(2), .github(1) |  |
| `59afd3a20` | 2026-07-17 | Anbeeld | fix KVarN determinism and KLD baselines | 8 | tests/(4), ggml/src/ggml-cuda(2), tools/other(2) |  |
| `72ee3215d` | 2026-06-18 | Anbeeld | Generalize KVarN dynamic stage ops | 8 | tests/(2), ggml/include(1), ggml/src/ggml-cpu(1) |  |
| `aa066a29d` | 2026-06-05 | Anbeeld | fix(kvarn): tighten runtime policy and API surface | 8 | src/(5), common/(2), other(1) |  |
| `e0663be27` | 2026-06-01 | Anbeeld | Improve DFlash multi-slot shared drafting | 8 | docs(2), src/(2), common/(1) |  |
| `0df2b4051` | 2026-05-25 | Anbeeld | server: stabilize dflash multi-slot state | 8 | src/(2), tests/(2), tools/server(2) |  |
| `dade9d3b5` | 2026-05-21 | Anbeeld | sync upstream Hadamard rotation plumbing | 8 | ggml/src/ggml-cpu(3), src/(3), ggml/include(1) |  |
| `3109a0bfa` | 2026-05-13 | Anbeeld | Refine profit baseline reprobes | 8 | common/(2), docs(2), tests/(2) |  |
| `a45cddaf6` | 2026-04-26 | spiritbuun | GPU cross-attention ring buffer for DFlash speculative decoding | 8 | src/(3), ggml/src/ggml-cuda(2), common/(1) |  |
| `c6f7eef25` | 2026-07-18 | Anbeeld | args: remove dflash speculative alias | 7 | other(4), common/(1), docs(1) |  |
| `28005442a` | 2026-06-04 | Anbeeld | ci: consolidate release workflow | 7 | .github(3), other(2), scripts/(2) |  |
| `1a93ab786` | 2026-06-02 | Anbeeld | server: make DFlash profit DM adaptive | 7 | docs(2), tests/(2), tools/server(2) |  |
| `507c1f7cc` | 2026-05-31 | Anbeeld | Align MTP server lifecycle and clean diagnostics | 7 | common/(3), src/(2), tests/(1) |  |
| `b839efd72` | 2026-05-26 | Anbeeld | Pin DFlash target output for explicit draft device | 7 | common/(2), docs(1), other(1) |  |
| `3d76953c9` | 2026-05-25 | Anbeeld | server: stabilize recurrent dflash overlap | 7 | common/(2), tests/(2), tools/server(2) |  |
| `8985e86cb` | 2026-05-23 | Anbeeld | mtmd: preserve Gemma3 full image chunks | 7 | tools/other(4), tools/server(2), tests/(1) |  |
| `44a8bd698` | 2026-05-19 | Anbeeld | Support upstream DFlash drafter GGUF schema | 7 | src/(5), tests/(2) |  |
| `431b98e5d` | 2026-05-18 | Anbeeld | dflash: add categorized profile logging | 7 | src/(4), common/(1), tests/(1) |  |
| `3450a8275` | 2026-04-15 | spiritbuun | C.0: Add SuffixDecoding model-free speculative decoding | 7 | common/(7) |  |
| `d29fb0b37` | 2026-03-28 | spiritbuun | feat: turbo4 4-bit PolarQuant - drop QJL, 16 Lloyd-Max centroids | 7 | ggml/src/ggml-cuda(3), ggml/src/other(2), other(1) |  |
| `b8a4a8532` | 2026-08-21 | SubSir | Revert draft sampling in rejection sampling | 6 | common/(4), examples/(1), tools/server(1) |  |
| `fc4c83995` | 2026-08-20 | Anbeeld | fix(server): detect pathological loops in visible output | 6 | tools/server(4), tests/(2) |  |
| `1fd04c99a` | 2026-07-11 | Anbeeld | Restructure BeeLlama reference documentation | 6 | docs(5), other(1) |  |
| `71e2fe4a4` | 2026-06-18 | Anbeeld | Fix Vulkan lambda captures, add backend shape asserts, unaligned tests, clean state comments | 6 | src/(2), ggml/src/ggml-cpu(1), ggml/src/ggml-cuda(1) |  |
| `adde280f9` | 2026-06-11 | Anbeeld | Add KVarN profiling instrumentation | 6 | ggml/src/ggml-cuda(3), tools/server(2), tests/(1) |  |
| `1761dfdd5` | 2026-06-05 | Anbeeld | Fix DFlash KVarN visible context sizing | 6 | tests/(3), tools/server(2), other(1) |  |
| `4f5703a2f` | 2026-06-05 | Anbeeld | ci: finalize v0.3.2 release workflow | 6 | .github(2), docs(2), other(1) |  |
| `f86953c3c` | 2026-06-03 | Anbeeld | Improve release Docker links and CI caching | 6 | other(5), .github(1) |  |
| `e536432d8` | 2026-06-01 | Anbeeld | Harden DFlash reduced logits and replay guards | 6 | common/(2), src/(2), ggml/src/ggml-cuda(1) |  |
| `37067063c` | 2026-05-27 | Anbeeld | DFlash: harden tensor-split/meta capture and avoid unsafe multi-GPU fallback | 6 | src/(2), common/(1), ggml/src/ggml-cuda(1) |  |
| `25b8b0ed2` | 2026-05-24 | Anbeeld | dflash: default slots to server parallelism | 6 | common/(3), docs(1), tests/(1) |  |
| `285933ed2` | 2026-05-22 | Anbeeld | v0.2.0 | 6 | docs(4), other(2) |  |
| `7c04e57de` | 2026-05-18 | Anbeeld | Add DFlash verifier diagnostics | 6 | src/models(3), src/(2), tools/server(1) |  |
| `5d770baf7` | 2026-04-20 | spiritbuun | SD-069: GPU-resident tape eliminates eval callback sync overhead (+17%) | 6 | src/(3), common/(1), examples/(1) |  |
| `5bc26d992` | 2026-04-08 | Georgi Gerganov | gemma : perform per-layer projections in the first layer (#21612) | 6 | src/(3), src/models(3) |  |
| `1f0217246` | 2026-03-25 | TheTom | fix: restore inverse rotation in dequant - PPL 6.19 (1.2% of q8_0) #31 #30 | 6 | src/(5), ggml/src/ggml-metal(1) |  |
| `bbc77c7a0` | 2026-07-28 | Anbeeld | Fix KVarN rollback failures and compact-tail reprocessing | 5 | src/(4), tests/(1) |  |
| `f9967bb52` | 2026-07-27 | Anbeeld | ci: seed version caches from predecessor branches | 5 | .github(3), tests/(2) |  |
| `94605e9fe` | 2026-07-27 | Anbeeld | ci: seed version caches from predecessor branches | 5 | .github(3), tests/(2) |  |
| `51fa5c14f` | 2026-07-26 | Anbeeld | hip: harden KVarN routes after release audit | 5 | ggml/src/ggml-cuda(3), tests/(2) |  |
| `7b982f526` | 2026-07-19 | Anbeeld | perplexity: restore upstream KLD baseline format | 5 | tests/(3), tools/other(2) |  |
| `4d13a2d36` | 2026-07-16 | Anbeeld | Document exact-tail safety and compatibility semantics | 5 | docs(5) |  |
| `adcf9e647` | 2026-07-11 | Anbeeld | Remove unclassified upstream merge residue | 5 | other(2), examples/(1), tools/other(1) |  |
| `82f7b5b3a` | 2026-06-23 | Anbeeld | Enable KVarN for single-stream Gemma SWA caches | 5 | src/(2), common/(1), docs(1) |  |
| `12f256a30` | 2026-06-06 | Anbeeld | fix(ci): ccache bugs A/B/C and bump SYCL compute runtime to 26.x | 5 | .github(3), other(2) |  |
| `84d4f1df6` | 2026-06-06 | Anbeeld | Fix flat DFlash recurrent rollback allocation | 5 | tests/(2), tools/server(2), common/(1) |  |
| `e72710983` | 2026-05-26 | Anbeeld | Fix MTP and DFlash speculative separation | 5 | common/(2), other(1), tests/(1) |  |
| `94b4e5f4a` | 2026-05-25 | Anbeeld | docs: add v0.3.0 changelog | 5 | docs(2), other(1), tools/other(1) |  |
| `43c14a631` | 2026-05-24 | Anbeeld | server: suppress leading thinking syntax in streamed titles | 5 | tests/(2), tools/server(2), common/(1) |  |
| `2970cdf7b` | 2026-05-18 | Anbeeld | Fix reduced sampler reasoning-end forcing | 5 | common/(3), tests/(2) |  |
| `f1d9677cf` | 2026-05-16 | Anbeeld | Revert DFlash replay stream changes from op-table fix | 5 | ggml/src/ggml-cuda(2), ggml/include(1), src/(1) |  |
| `9a388b52e` | 2026-04-25 | spiritbuun | DFlash + checkpoint compatibility: flush_prefill, ring state save/restore | 5 | common/(2), tools/server(2), other(1) |  |
| `1fc4d0e7b` | 2026-04-25 | spiritbuun | B2.6: multi-seq verify batching - remove force_split_seq | 5 | src/(3), src/models(2) |  |
| `2cc599bbc` | 2026-04-24 | spiritbuun | B2.2 (option A): pass dflash_n_slots through llama_context_params at init | 5 | common/(2), other(1), src/(1) |  |
| `5a95ed380` | 2026-04-09 | Piotr Wilkin (ilintar) | vocab: add gemma4 tokenizer tests, fix edge case (#21534) | 5 | other(3), src/(1), tests/(1) |  |
| `764c686b0` | 2026-04-01 | spiritbuun | feat: TCQ error autocorrelation measurement - errors are iid, not correlated | 5 | other(2), ggml/src/ggml-cuda(2), scripts/(1) |  |

## Сводка по областям

| Область (префикс пути) | Кол-во коммитов | Примеры SHA |
|---|---|---|
| src/ | 152 | c6c438f4e 11f45ed34 2474373ec 5ecbe1ac1 |
| ggml/src/ggml-cuda | 149 | c7a6b18d4 d1a522fc8 7ea40ee98 fde7797a2 |
| other | 125 | 3746fb185 fb500579d 005128c49 287da7cd0 |
| tests/ | 90 | 2a5e1bc9e a7417e85b 2d7a40c6b 1ce739a01 |
| common/ | 90 | f5a7ec15d c184d9a7a b8a4a8532 c314bb10d |
| .github | 54 | cd3c41e73 ed8db4abf 5a357925e e66f4d4f2 |
| src/models | 39 | 2f3923bc8 f7aadef09 cf095c834 bfaa12563 |
| docs | 39 | 0fb10f6b4 9084f1bed 5eaba174c 41d41b0ce |
| tools/server | 33 | fc4c83995 fc9ab2bd6 83d4173d8 8fb188a6f |
| ggml/src/ggml-metal | 16 | 6d019c3b8 413b33253 aa6a3a180 654647aac |
| ggml/src/other | 15 | 2525c60cf 634ffc9be 83429d46a 9572674dc |
| ggml/src/ggml-cpu | 11 | ad33bb4b2 e4cfc12f8 0ef8c42f4 9f268fb03 |
| tools/other | 9 | ab8a22e5b 8985e86cb c5eefecf8 4db14be0a |
| examples/ | 8 | 1006e84f0 0e4945217 edecf9699 d16a50d3b |
| scripts/ | 6 | e9eb7eec1 bc4558e60 4a834dd7a 1ae630920 |
| ggml/src/ggml-vulkan | 4 | 2c919f2ce a620cbd48 b639edb1d 12a169b8f |
| gguf-py | 3 | 64f765f5a 64885bc9d 9d4a73a49 |
| cmake | 2 | d237e4e47 89364abbe |
| ggml/include | 2 | e1cf4bf52 7cae8affd |
| conversion/ | 1 | e0181cd6c |

## Ограничения сбора
- Сеть требовала обхода неверного HTTP-прокси (HTTP_PROXY/HTTPS_PROXY в окружении указывали на
  недоступный адрес 191.101.126.19:50101). Операции выполнялись со сброшенными прокси-переменными;
  после этого клонирование и fetch upstream/master прошли успешно. Данных не выдумывал.
- Клонирование с --filter=blob:none прошло без падения фильтра (partial clone). Для git log с
  --name-only blobs подтягивались из promisor-remote по требованию.
- upstream/master на момент сбора: upstream/master = 95ef7fc16054e63b427a3ef00188e055ef7586d8 (2026-09-03). Сравнение велось именно против этого ref
  (git log upstream/master..HEAD). Ahead/Behind считаются относительно него.
- Классификация механическая: область = верхняя директория первого/самого частого изменённого пути;
  перф-маркеры = слова из subject по границам слов (регистронезависимо). Это черновой фильтр для
  последующего ручного анализа, не вердикт.
- "Ключевые директории" показывают до 3 верхних путей с числом файлов; полный список путей -
  в промежуточном файле D:/dev/llama_compiling/_forks_opt/beellama_classified.md (вне дерева проекта).
- Маркеры архитектуры/локальных тем искались только по код-файлам (.cpp/.h/.cu/.py/.metal/...),
  markdown (README/AGENTS/NOTES) исключён из доказательной базы.

