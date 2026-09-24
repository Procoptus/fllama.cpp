# Реестр: все 144 коммита fewtarius/CachyLLama (вне upstream)

Источник: `D:\dev\llama_compiling\_forks_opt\cl_full_log.txt` (144 non-merge коммита,
HEAD `8dca17b4d`, merge-base `3466812d1`). ID = `CL-<sha7>`, ссылка на коммит по ID.
Причины N/A раскрыты в [04-not-applicable.md](04-not-applicable.md),
применимое детализировано в [02-applicable.md](02-applicable.md).

Итог: PORTED 3 | CUDA-APPLICABLE 6 | UPSTREAM 1 | BLOCKED 48 | N/A 86
(POSIX-адаптация под MSVC/Windows: unistd->_io/fcntl, mmap->_open_osfhandle+MapViewOfFile,
madvise-нет, host-RAM через GlobalMemoryStatusEx. Сборка и unit-тест SSD-кэша проходят.
Ветка `ssd-kv-cache`, фиксы группы BLOCKED/SSD-cache теперь переносимы, но пока не перенесены.)
N/A breakdown без изменений: Vulkan 26, ROCm 4, Metal/macOS 2, чужие арх-фичи 4, Linux-only 15, неиспользуемые фичи 18, fork-internal 16, ISWA-путь 1.
(Vulkan 26, ROCm 4, Metal/macOS 2, чужие арх-фичи 4, Linux-only 15, неиспользуемые фичи 18, fork-internal 16, ISWA-путь 1).

## 1. CUDA-APPLICABLE - применимо нам, нужно портировать (9)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [CL-a657590ae](https://github.com/fewtarius/CachyLLama/commit/a657590aecf40532fd8fdd6e111786ee8614270c) | 2026-08-27 | qwen4exp: implement true sparse gather for QSA attention | Наш случай 1:1. В нашем дереве нет gather-ОПЕРАЦИИ для KV: после `ggml_top_k` ([qwen4exp.cpp:684](src/models/qwen4exp.cpp:684)) attention считается плотной маской по всем n_kv ([qwen4exp.cpp:695](src/models/qwen4exp.cpp:695)); слово gather - только в комментариях. Приоритет 1 |
| [CL-f658fc5af](https://github.com/fewtarius/CachyLLama/commit/f658fc5af98f404789b6200160892b9ebc87b8e4) | 2026-08-28 | server: skip host-memory prompt cache round-trip when n_parallel <= 1 | У нас ровно parallel=1 + cache-ram 3000; guard'а нет ([server-context.cpp:1625-1646](tools/server/server-context.cpp:1625)). Приоритет 2 |
| [CL-b83d23022](https://github.com/fewtarius/CachyLLama/commit/b83d23022d1252c1c005e95e44db2cdc9fd154f8) | 2026-08-22 | fix: seq_rm in recurrent memory should fall through on rollback failure | Баг жив в нашем дереве ([llama-memory-recurrent.cpp:161](src/llama-memory-recurrent.cpp:161)). Приоритет 3 |
| [CL-67f82f589](https://github.com/fewtarius/CachyLLama/commit/67f82f589f8358104885d4815ca3effa764b7540) | 2026-08-12 | ggml: cut backend splits on the input constant, not the grown capacity | Ratchet-баг жив ([ggml-backend.cpp:1341](ggml/src/ggml-backend.cpp:1341)); у нас CPU-оффлоад MoE -> splits. Приоритет 4 |
| [CL-024dff234](https://github.com/fewtarius/CachyLLama/commit/024dff234d40871771926c6b152b0fa8d19d9dfe) | 2026-07-07 | fix(server): evict highest-pos_min checkpoint instead of insertion-oldest | У нас evict по oldest front ([server-context.cpp:2317-2322](tools/server/server-context.cpp:2317)), -ctxcp 3. Требует адаптации. Приоритет 5 |
| [CL-6a0db500c](https://github.com/fewtarius/CachyLLama/commit/6a0db500ca1058e06a232c02c50eb5df56b0d151) | 2026-06-26 | feat(ssd): SSD-backed KV cache with hybrid model checkpoint restore | **PORTED 2026-09-11** (ветка `ssd-kv-cache`): kv-ssd-cache + page-manager в дереве, флаги `--cache-ssd*`, тест `test-kv-ssd-user-isolation` проходит |
| [CL-56dca0825](https://github.com/fewtarius/CachyLLama/commit/56dca0825e1d0b3a4b5f00a1fc1e59f2c6a900f8) | 2026-06-14 | feat(ssd-cache): add global system prompt KV cache | **PORTED 2026-09-11** (в составе SSD-порта) |
| [CL-cc60f8912](https://github.com/fewtarius/CachyLLama/commit/cc60f89123cc5c800717bed1b9c5892d7aac12c4) | 2026-06-14 | feat(server): wire global system prompt KV cache into server lifecycle | **PORTED 2026-09-11** (wiring в server-context.cpp в дереве) |
| [CL-022c16ed5](https://github.com/fewtarius/CachyLLama/commit/022c16ed51f271ce016f6be5714fe23924c24c06) | 2026-08-23 | server: detect conversation boundaries via conv_hash + K-shift NULL buffer crash | Defensive K-shift guard уместен ([llama-kv-cache.cpp:1526](src/llama-kv-cache.cpp:1526) - GGML_ASSERT без null-guard). conv_hash часть = BLOCKED. Приоритет 8 |

## 2. UPSTREAM - эквивалент уже есть (1)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [CL-4db9548a8](https://github.com/fewtarius/CachyLLama/commit/4db9548a8c421da633e1124b4d4110b82306803e) | 2026-07-03 | fix(kv-cache): disable shift/cache-reuse for IMROPE models (n_pos_per_embd > 1) | Guard уже в нашем дереве ([llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189)). Детали в [03-already-upstream.md](03-already-upstream.md) |

## 3. BLOCKED - применимо только после портирования подсистемы-базы (48)

Подсистемы-базы: SSD KV cache (CL-6a0db500c) + sys-prompt cache (CL-56dca0825/CL-cc60f8912) -
**перенесены 2026-09-11**, ветка `ssd-kv-cache`; фиксы группы "SSD cache / cold-start" разблокированы
и теперь переносимы (пока не перенесены). API `seq_rm_attn_only` по-прежнему отсутствует -
фиксы на нём остаются BLOCKED.

| ID | Дата | Предмет | Блокирующая база |
|---|---|---|---|
| [CL-54e51d11f](https://github.com/fewtarius/CachyLLama/commit/54e51d11ff832442423e6fd698c3d6ef5911ac49) | 2026-08-28 | qwen4exp: keep indexer cache in sync with attention under seq_rm_attn_only | seq_rm_attn_only |
| [CL-5dd48c54b](https://github.com/fewtarius/CachyLLama/commit/5dd48c54be5f7387538f791be4388f6eb1e83246) | 2026-07-21 | fix(hybrid): clear stale mem_recr positions in seq_rm_attn_only (closes #8) | seq_rm_attn_only |
| [CL-757361a70](https://github.com/fewtarius/CachyLLama/commit/757361a707edd6433c7a647bfd0982cc46cb90bf) | 2026-07-26 | fix(server): resolve "Invalid input batch" on multi-turn conversations (issue #8) | deferred ckpt + seq_rm_attn_only |
| [CL-96937d47c](https://github.com/fewtarius/CachyLLama/commit/96937d47c799787c37e855d4a0befe863395ab01) | 2026-06-25 | fix(ssd): defer final checkpoint to after first token | checkpoint ring |
| [CL-da9a6da82](https://github.com/fewtarius/CachyLLama/commit/da9a6da82b6a3d43361d915bbac7b4d5d6d022a6) | 2026-08-18 | server: off-by-one in deferred_create_final_checkpoint + ... | checkpoint ring |
| [CL-906bb6fcb](https://github.com/fewtarius/CachyLLama/commit/906bb6fcb03c9d0b5e77456b24e160dba13b234e) | 2026-08-20 | server: fix stale eviction policy comments + clarify ring buffer log format | checkpoint ring |
| [CL-dbedc90ec](https://github.com/fewtarius/CachyLLama/commit/dbedc90ecb08940e9e421da8036eaaf7192fff93) | 2026-08-17 | server: ring buffer recycling, SWA-skip, memory budget scaling | checkpoint ring |
| [CL-6af265fa1](https://github.com/fewtarius/CachyLLama/commit/6af265fa1f125ae54658be6a2bc2bacaf2fe769f) | 2026-08-20 | server: fix pos_min gate + improve ring buffer log messages | checkpoint ring |
| [CL-2f1fc4b12](https://github.com/fewtarius/CachyLLama/commit/2f1fc4b12d7303e3814453e29a13efaf9b7f001a) | 2026-07-07 | server: add probe logging to do_reset/LCP path | checkpoint ring |
| [CL-cea45f422](https://github.com/fewtarius/CachyLLama/commit/cea45f4222e1eea5c2c68388d7e75904a65b6778) | 2026-07-27 | fix(server): relax ckpt filter for non-SWA hybrid models, demote probe/telemetry logs | checkpoint ring |
| [CL-31a062d52](https://github.com/fewtarius/CachyLLama/commit/31a062d52831b5c9c9b13b2207295e7ae098cd18) | 2026-07-18 | fix(ssd): gate near-prompt-end checkpoint behind --checkpoint-near-end | checkpoint ring |
| [CL-1585f98b3](https://github.com/fewtarius/CachyLLama/commit/1585f98b3b400ec6e0cd3466d4fc9cccffb078ce) | 2026-08-18 | server: fix prompt cache metadata/source bugs; add stable-prefix gate | prompt/sys cache |
| [CL-835013b39](https://github.com/fewtarius/CachyLLama/commit/835013b3927b552dde508e440acaeedab1501bea) | 2026-06-26 | feat(ssd): system prompt cache + hybrid model warm restart fixes | sys-prompt cache |
| [CL-70ceb7b14](https://github.com/fewtarius/CachyLLama/commit/70ceb7b141f0e875b24203186c5ea450e7d0c7c6) | 2026-06-14 | fix(server): enable system prompt cache for hybrid models | sys-prompt cache |
| [CL-926f343fa](https://github.com/fewtarius/CachyLLama/commit/926f343fa655f620b380468a34483c80c4f38855) | 2026-06-14 | fix(server): align sys_cache compat hash with page_manager | sys-prompt cache |
| [CL-de62cdfbd](https://github.com/fewtarius/CachyLLama/commit/de62cdfbd5def9104a396975b0679eb90fd23f5e) | 2026-06-30 | fix(server): system prompt cache save/load flag mismatch and slot context guard | sys-prompt cache |
| [CL-181b6a3ca](https://github.com/fewtarius/CachyLLama/commit/181b6a3ca14e4642468bfbbb4d23c9092c9f4bb9) | 2026-07-19 | fix(sys-cache): add prefix-match fallback for do_reset recovery | sys-prompt cache |
| [CL-17d619fd0](https://github.com/fewtarius/CachyLLama/commit/17d619fd0379efbc5f6d525bb0c4fea9bb2a18ae) | 2026-07-19 | fix(server): fall back to system prompt cache after do_reset on warm slots | sys-prompt cache |
| [CL-dffcd625d](https://github.com/fewtarius/CachyLLama/commit/dffcd625ddbe79ca50893a3df8c857dd1d6db89b) | 2026-06-29 | fix(ssd): restore system prompt cache init lost in rebase | sys-prompt cache |
| [CL-521db2771](https://github.com/fewtarius/CachyLLama/commit/521db2771dd5ca3876ca5a94517e4d0f428fa999) | 2026-06-20 | fix(boundary): detect system prompt boundary for GLM/Gemma templates | sys-prompt cache |
| [CL-59da9e100](https://github.com/fewtarius/CachyLLama/commit/59da9e10004cc454d0062d1dfbbd5e0e7b1756f8) | 2026-08-29 | kv-ssd: v4 format + atomic writes + checksum, plus routing isolation tests | SSD cache |
| [CL-105889b46](https://github.com/fewtarius/CachyLLama/commit/105889b4645570e0f249247bcf37c9fe0c7c31de) | 2026-08-22 | server: accept partial LCP matches in SSD cache to fix cache hit collapse after agent trims | SSD cache |
| [CL-2697d1c57](https://github.com/fewtarius/CachyLLama/commit/2697d1c5712d8dd387a3ebd846fcbecd27a3fbe1) | 2026-07-23 | fix(ssd-cache): score continuation overlap against stored prefix length | SSD cache |
| [CL-86ba3b1b8](https://github.com/fewtarius/CachyLLama/commit/86ba3b1b8f6105b7be4fdc2ce31126687dc9042a) | 2026-07-23 | server : document SSD checkpoint verification | SSD cache |
| [CL-1c09c4400](https://github.com/fewtarius/CachyLLama/commit/1c09c4400650fdc09911bf4da89c604e41d09fc2) | 2026-07-23 | server : verify SSD checkpoint token prefixes | SSD cache |
| [CL-70ecaf4c5](https://github.com/fewtarius/CachyLLama/commit/70ecaf4c5ee47ca7885a3058efe8947018cbb0ca) | 2026-07-22 | fix(kv-cache-dsv4): remove broken seq_rm safety gate | SSD/seq_rm path (DSV4-only файл) |
| [CL-a08178e88](https://github.com/fewtarius/CachyLLama/commit/a08178e889d89539a90aa2e64b282abddac8f6ee) | 2026-08-10 | dsv4: fix rollback with multi-seq + recurrent test coverage | SSD/seq_rm path (DSV4-only файл) |
| [CL-a5fd2cd13](https://github.com/fewtarius/CachyLLama/commit/a5fd2cd13e4b280aad705aac6aebdfe9fe5b20ee) | 2026-07-21 | fix(kv-cache-dsv4): allow seq_rm truncation to post-restore LCP (closes #8) | SSD/seq_rm path (DSV4-only файл) |
| [CL-e21a83d32](https://github.com/fewtarius/CachyLLama/commit/e21a83d32291a8a8ebbf949421f5019159ccb34e) | 2026-07-21 | server : add --cache-ssd-cold-maxsize global byte cap | SSD cache |
| [CL-95165a175](https://github.com/fewtarius/CachyLLama/commit/95165a175c7847adb7f9328dad4195c5c22272ed) | 2026-07-18 | fix(ssd): respect --cache-ssd-hot-ram and --cache-ssd-warm-ram caps | SSD cache |
| [CL-d9bbcfa42](https://github.com/fewtarius/CachyLLama/commit/d9bbcfa423332348373125456dd57650dfaffa23) | 2026-06-03 | feat(server): add --cache-ssd-hot-ram and --cache-ssd-warm-ram CLI flags | SSD cache |
| [CL-a4e459ef0](https://github.com/fewtarius/CachyLLama/commit/a4e459ef06995e1f7bb882be80e53485ff734a17) | 2026-06-30 | feat(ssd): add --cache-ssd-no-fsync flag for async checkpoint writes | SSD cache |
| [CL-43d781ac0](https://github.com/fewtarius/CachyLLama/commit/43d781ac08636b4531214816b6c89df9f50afcb3) | 2026-06-01 | ssd-cache: add kernel readahead for cold checkpoint prefetch | SSD cache |
| [CL-3162f6155](https://github.com/fewtarius/CachyLLama/commit/3162f61556118aa5097c72b8250ca130ad54580e) | 2026-07-05 | Mark: server: persist context checkpoints in a sidecar file for slot save/restore | SSD cache |
| [CL-8b2cf6c66](https://github.com/fewtarius/CachyLLama/commit/8b2cf6c668a886a9e5bb3519c926b67484db2593) | 2026-07-05 | 3rd Iteration: Make SSD error messages more descriptive and fix large context failure (#3) | SSD cache |
| [CL-9d28796fd](https://github.com/fewtarius/CachyLLama/commit/9d28796fd5c87dce5827f4b75a5118e1ac50ce40) | 2026-07-06 | fix(ssd): fix thread safety, hex format portability, and POSIX log noise | SSD cache |
| [CL-bce3f53fd](https://github.com/fewtarius/CachyLLama/commit/bce3f53fd9e799d4ce2fe80b00583f68bd915f30) | 2026-07-05 | fix(ssd): deferred checkpoint skipped for dense models on short prompts | SSD cache |
| [CL-a185f9d84](https://github.com/fewtarius/CachyLLama/commit/a185f9d84a8d1db2eeaf229acfbaf53d58d20065) | 2026-08-16 | server: fix hallucination on cold-start from deferred checkpoint | SSD cold-start |
| [CL-ef67a59ca](https://github.com/fewtarius/CachyLLama/commit/ef67a59ca62b3cd7ce3c281bcb461a345b8235f0) | 2026-06-30 | server: fix n_past not set after SSD cold-start restore for hybrid models | SSD cold-start |
| [CL-83af232aa](https://github.com/fewtarius/CachyLLama/commit/83af232aad85daa62175ba6220c1200f67089b8a) | 2026-06-27 | fix(ssd): restore cold-start max_n_tokens guard lost in rebase | SSD cold-start |
| [CL-a461a1fce](https://github.com/fewtarius/CachyLLama/commit/a461a1fcebafa2a90ad635ba749c5a1ae1c4be8d) | 2026-06-28 | Asad Ali Bhatti: fix(ssd): restore KV cells under current slot's seq_id on cross-slot restore | SSD cold-start |
| [CL-602bf12df](https://github.com/fewtarius/CachyLLama/commit/602bf12df5e5770b06055fdbb9485c769d2d70ec) | 2026-06-30 | fix(server): restore per-user isolation and hybrid safety lost in rebase | SSD cold-start |
| [CL-9b06c5a95](https://github.com/fewtarius/CachyLLama/commit/9b06c5a95da4cf340fe43da558ef5e14cbcaeaa3) | 2026-06-27 | fix(ssd): extend seq_rm_attn_only guard to in-memory checkpoint restores | SSD + seq_rm_attn_only |
| [CL-c8ead677a](https://github.com/fewtarius/CachyLLama/commit/c8ead677a7fe42fb0a67e6e866fb254cc338e9fd) | 2026-07-08 | fix(ssd): pass dest_seq_id on continuation path (part 2 of cross-slot restore fix) | SSD cold-start |
| [CL-f1921da73](https://github.com/fewtarius/CachyLLama/commit/f1921da730f88307dcf97c629ce6a441e5858693) | 2026-05-30 | SSD cache: trust same-conversation checkpoints with full prefix match | SSD cache |
| [CL-afa4323d4](https://github.com/fewtarius/CachyLLama/commit/afa4323d47e619050974279e6309408308fe26ed) | 2026-05-30 | SSD cache: fix cold-start continuation matching and hybrid model validation | SSD cache |
| [CL-3e38e4a79](https://github.com/fewtarius/CachyLLama/commit/3e38e4a799e1b79eec77bf7ec864e8e9390ce10c) | 2026-05-30 | fix(ssd): enable cold-start SSD restore for hybrid models with LCP validation | SSD cache |
| [CL-cedeacccc](https://github.com/fewtarius/CachyLLama/commit/cedeacccc7b7af3f96335e889d0df597380a8bbf) | 2026-06-10 | fix(ssd-cache): set out_overlap for same-conversation cold-start matches | SSD cache |

## 4. N/A-VULKAN - Vulkan-ядра, шейдеры, тюнинг, доки (26)

Нам не применимы: сборка без Vulkan (CUDA/Windows). Числа эффекта автора - Strix Halo/RADV.

| ID | Дата | Предмет |
|---|---|---|
| [CL-a17e0a9e1](https://github.com/fewtarius/CachyLLama/commit/a17e0a9e136bade8b483ac7d58513989769c59a0) | 2026-08-27 | vulkan: widen topk_f32 pipelines (11 -> 14) for QSA indexer gather |
| [CL-c22b79e17](https://github.com/fewtarius/CachyLLama/commit/c22b79e174d4318b43896c68634f9454ef8576c6) | 2026-08-27 | fix(merge): resolve shader gen duplicate and test archs syntax after upstream merge |
| [CL-887d408ba](https://github.com/fewtarius/CachyLLama/commit/887d408ba905b54d3db6f9f35857e3647055a5e7) | 2026-08-13 | vulkan: wire coopmat lightning indexer dispatch (prefill + decode paths) |
| [CL-402d5a7e7](https://github.com/fewtarius/CachyLLama/commit/402d5a7e7bf14aeab97d2b9f002306cbaa0f72fe) | 2026-08-12 | vulkan: fix Lightning Indexer (108/108) + Strix Halo prefill/MoE perf |
| [CL-85d281a23](https://github.com/fewtarius/CachyLLama/commit/85d281a23695c8ea22e7d5f9c18be888282e5e61) | 2026-08-12 | vulkan: add C2 DSV4 sparse FA + coopmat lightning indexer shaders (gaetan-puleo) |
| [CL-5f12949af](https://github.com/fewtarius/CachyLLama/commit/5f12949af4a323a88273346d40d2d961c5da1d7d) | 2026-08-12 | vulkan: wire CONCAT_TRANSPOSE dispatch (env-gated, default ON) (nathanw1014) |
| [CL-5aed1f989](https://github.com/fewtarius/CachyLLama/commit/5aed1f989ba73f65313c5327382ab3d3ffb425c2) | 2026-08-12 | vulkan: mmid row-list prepass for grouped-GEMM redesign (nathanw1014) |
| [CL-9902e67a0](https://github.com/fewtarius/CachyLLama/commit/9902e67a0753d468bd1d9a87676cba4f9b4a6e64) | 2026-08-12 | vulkan: pin a 32-wide subgroup for coopmat1 FA where narrowing is free (nathanw1014) |
| [CL-e9691599a](https://github.com/fewtarius/CachyLLama/commit/e9691599a95ad184da3797e5db1b2fe11bf3cbd1) | 2026-08-12 | vulkan: store coopmat1 FA Psh query-major so the GEMM2 A load vectorizes (nathanw1014) |
| [CL-481c195a8](https://github.com/fewtarius/CachyLLama/commit/481c195a85ea6732ddd3788512b99bcfc1fed01c) | 2026-08-12 | vulkan: hoist the coopmat1 FA P-fragment load out of the hsv_tile loop (nathanw1014) |
| [CL-33cc3c520](https://github.com/fewtarius/CachyLLama/commit/33cc3c520a54c91c0607bcb6c46e9a1e19554c70) | 2026-08-12 | vulkan: extend FA dequant-once to q4_0, q4_1, q5_0, q5_1 KV (nathanw1014) |
| [CL-404732f8c](https://github.com/fewtarius/CachyLLama/commit/404732f8c094c0a174f4439e549004f675247646) | 2026-08-12 | vulkan: contiguize strided f16 KV for FA prefill, opt-out toggle (nathanw1014) |
| [CL-e36a9872a](https://github.com/fewtarius/CachyLLama/commit/e36a9872a6b8caf22c67567253f57d14fb027865) | 2026-08-12 | vulkan: extract ggml_vk_fa_kv_native() for FA native-type classification (nathanw1014) |
| [CL-b3a9e851e](https://github.com/fewtarius/CachyLLama/commit/b3a9e851e1971d6ac40e713da7640990033f9470) | 2026-08-12 | vulkan: add concat_transpose shader for delta-net dim-0 concat (nathanw1014) |
| [CL-ae5458776](https://github.com/fewtarius/CachyLLama/commit/ae5458776ad4af3808fe9193d2b3573c3f5f3381) | 2026-08-12 | vulkan: scale the FA MMQ dot product in fp32 before narrowing (nathanw1014) |
| [CL-0a506ca1c](https://github.com/fewtarius/CachyLLama/commit/0a506ca1caa920a2a0cd72e82f8b245859b259a0) | 2026-08-12 | vulkan: bound command buffers by memory traffic, not just flops (nathanw1014) |
| [CL-3cb0dc761](https://github.com/fewtarius/CachyLLama/commit/3cb0dc761993cc25669e9c4d3730d3c6f91f10dd) | 2026-08-12 | vulkan: flush pending compute ctx before perf logger timestamps (nathanw1014) |
| [CL-d84ea1ca2](https://github.com/fewtarius/CachyLLama/commit/d84ea1ca2efe184bb1d923fe3c7ad681e778e66c) | 2026-08-04 | vulkan : add BF16 K-cache support to Lightning Indexer fused op |
| [CL-760e06e63](https://github.com/fewtarius/CachyLLama/commit/760e06e63f8169fd53c18bdd6581ad8f9e572da4) | 2026-08-04 | vulkan: add DeepSeek-V4 Lightning Indexer fused op with init-order fix |
| [CL-1b7ea8149](https://github.com/fewtarius/CachyLLama/commit/1b7ea814934a8cf6a8b21968f7062fa101ca91ab) | 2026-08-02 | Kevin Hopper: vulkan: add DeepSeek-V4 hyper-connection fused ops (DSV4_HC_COMB/PRE/POST) |
| [CL-a16163278](https://github.com/fewtarius/CachyLLama/commit/a1616327810895158b1f291eb053b52269391886) | 2026-08-02 | Kevin Hopper: vulkan: tiled transpose for 0<->2 permuted CONT |
| [CL-c0407cd21](https://github.com/fewtarius/CachyLLama/commit/c0407cd219454e3f27f460bbef3e85ebe5593d8e) | 2026-07-23 | vulkan : restore FA dequant+transpose safety gates from upstream #25494 |
| [CL-bd4f2875b](https://github.com/fewtarius/CachyLLama/commit/bd4f2875b177abd3ada0bdd39318293cdeb3b4d5) | 2026-07-19 | vulkan : dequant q8_0 KV once in coopmat1 with CachyLLama memory gate |
| [CL-1c19480da](https://github.com/fewtarius/CachyLLama/commit/1c19480dafb6d705613d5492bdeaed015c8a89d9) | 2026-06-11 | fix(vulkan): auto-lower nodes_per_submit for APU/iGPU + GGML_VK_NODES_PER_SUBMIT override |
| [CL-15e399d01](https://github.com/fewtarius/CachyLLama/commit/15e399d0176edae69d884133ffcce55e5766057b) | 2026-07-20 | docs(vulkan): add CEZANNE_NOTES.md (gfx90c sweep results on zaphod) |
| [CL-0da40ed6c](https://github.com/fewtarius/CachyLLama/commit/0da40ed6cd40e787848dc1333ba9711a821a2fc6) | 2026-07-20 | docs(vulkan): add RDNA3 (Phoenix) tuning notes and nps override comment |

## 5. N/A-ROCm - ROCm/HIP, RDNA3.5 (gfx1151) (4)

| ID | Дата | Предмет |
|---|---|---|
| [CL-5ff23cdf3](https://github.com/fewtarius/CachyLLama/commit/5ff23cdf339d58cf8e1a0a1f629dd9151b53c71d) | 2026-06-28 | perf(rocm): apply RDNA3.5 Strix Halo tuning from gaetan-puleo |
| [CL-71d1e8f2f](https://github.com/fewtarius/CachyLLama/commit/71d1e8f2fbf769e2b61d50742188794f4acf3303) | 2026-07-20 | fix(cuda): bump rdna3_5 MMQ I from 48 to 64 to unblock gfx1151 build |
| [CL-2b0c4dcfd](https://github.com/fewtarius/CachyLLama/commit/2b0c4dcfd68caeda50c89538cba57b438c9b08dc) | 2026-07-22 | fix(ggml-cuda): size MMQ per-thread sum[] for both dp4a and MMA layouts |
| [CL-d95602b56](https://github.com/fewtarius/CachyLLama/commit/d95602b56f0bc93b01094122110ee0ce0c0b1729) | 2026-07-20 | ci(self-hosted): add gfx1151 (Strix Halo) HIP smoke build |

Примечание: CL-2b0c4dcfd правит ggml-cuda, но сбой (HSA memory fault) существует только
на AMD/RDNA3.5 MMA-пути; на NVIDIA dp4a-размер корректен. См. [04-not-applicable.md](04-not-applicable.md) группа 2.

## 6. N/A-METAL - macOS (2)

| ID | Дата | Предмет |
|---|---|---|
| [CL-b782fb874](https://github.com/fewtarius/CachyLLama/commit/b782fb8749a318ca1c154a6e0e918923bcb6a171) | 2026-06-17 | build(macos): fix CachyLLama build on macOS |
| [CL-b6bf93092](https://github.com/fewtarius/CachyLLama/commit/b6bf93092c947bbf7fd43793eb6cb2deed7f27ae) | 2026-07-18 | fix(ssd): detect host RAM on macOS for auto-sizing |

## 7. N/A-ARCH - чужие модели/парсеры, UMA-специфика (4)

| ID | Дата | Предмет |
|---|---|---|
| [CL-7140f5fae](https://github.com/fewtarius/CachyLLama/commit/7140f5fae1d38e4aa1e8c773471830b171f14a02) | 2026-08-29 | ssd: subtract model footprint from auto-size budget on UMA APUs |
| [CL-bc499b060](https://github.com/fewtarius/CachyLLama/commit/bc499b060c2ceb55a9b426fb0b0e0bf41a90d59a) | 2026-08-10 | chat: fix DSML/PEG parser for DeepSeek-V4-Flash tool calls (2 fixes) |
| [CL-7caa42de1](https://github.com/fewtarius/CachyLLama/commit/7caa42de1fd10bcd8f9f6bdb72b0b396102bd1d6) | 2026-08-10 | chat: strip unclosed/orphaned DSML tokens in degenerate responses |
| [CL-af6fbcb2a](https://github.com/fewtarius/CachyLLama/commit/af6fbcb2a4e97771da66de1f7e06eb2023a17a97) | 2026-08-30 | server: fix Laguna tag parser greedily consuming nested arg_key as value |

## 8. N/A-LINUX-ONLY - madvise/mincore, /proc, Linux CI (15)

| ID | Дата | Предмет |
|---|---|---|
| [CL-3ebbc7ea4](https://github.com/fewtarius/CachyLLama/commit/3ebbc7ea43b4e7284e24a1b90d51bbee05d36f9d) | 2026-07-21 | feat(moe-offload): Phase 1 - per-token expert selection + madvise residency |
| [CL-2c8642383](https://github.com/fewtarius/CachyLLama/commit/2c86423831df3eeabd23e51b54bb93d8663c2659) | 2026-07-21 | feat(moe-offload): Phase 2 - R+F cache scoring + co-activation matrix |
| [CL-8edfad218](https://github.com/fewtarius/CachyLLama/commit/8edfad21805c22c8131eaec70456e548d81178db) | 2026-07-27 | fix(moe-residency): use MADV_FREE + larger cache to fix page-fault perf regression |
| [CL-f6c412a61](https://github.com/fewtarius/CachyLLama/commit/f6c412a612b986dcc9107e126f55de07323d9d79) | 2026-08-28 | moe-residency: fix MADV_FREE on MAP_SHARED + add observability |
| [CL-880d52c07](https://github.com/fewtarius/CachyLLama/commit/880d52c072a0afc6ccbb65cdafab3c08112fd89b) | 2026-07-21 | fix(moe-offload): use LLAMA_LOG_WARN for residency messages |
| [CL-bc00778cd](https://github.com/fewtarius/CachyLLama/commit/bc00778cdfda54433485dcb9dfc0e2822fd7ca4a) | 2026-07-21 | perf(moe-offload): bump defaults for higher hit rates |
| [CL-aa4644dc6](https://github.com/fewtarius/CachyLLama/commit/aa4644dc66bc9a5d9a25539f28691ec8e2e2632a) | 2026-07-21 | fix(moe-offload): fall back to computing top-K from F32 probs tensor |
| [CL-ebff3e87c](https://github.com/fewtarius/CachyLLama/commit/ebff3e87c3dedf0833991259f1a9871507561c71) | 2026-07-21 | feat(moe-offload): accept both ffn_moe_argsort and ffn_moe_topk tensors |
| [CL-668dc5150](https://github.com/fewtarius/CachyLLama/commit/668dc5150820c2cce1500867fbb4687f8d367c7c) | 2026-07-21 | fix(moe-offload): synchronize before reading argsort tensor data |
| [CL-b40698c7f](https://github.com/fewtarius/CachyLLama/commit/b40698c7f64f070ef84176a5d925f80ddeb2f12c) | 2026-07-26 | fix(merge): add --load-mode arg and fix moe-expert-residency check for load-mode refactor |
| [CL-34106161b](https://github.com/fewtarius/CachyLLama/commit/34106161be4b6c0e9f0c18a64ea5c694f6177eb8) | 2026-07-21 | docs: MoE expert residency subsystem |
| [CL-b69de8497](https://github.com/fewtarius/CachyLLama/commit/b69de84971815ba49a7dc57b062633f8c4369c9b) | 2026-06-01 | feat(server): add MoE expert activation tracking API |
| [CL-1ffbac42a](https://github.com/fewtarius/CachyLLama/commit/1ffbac42a424aa32065a628d4ad548b7443ffef9) | 2026-06-01 | feat: add MoE expert activation tracking API (Phase 1) |
| [CL-86855c4ae](https://github.com/fewtarius/CachyLLama/commit/86855c4ae9cf395c05b211aac9dee9ca83e5a688) | 2026-07-21 | fix(moe-offload): skip empty MoE routing tensors in track_expert_activations |
| [CL-a2b13f5ab](https://github.com/fewtarius/CachyLLama/commit/a2b13f5ab98766a168ec5f4a97ad5f37a82c5c66) | 2026-07-04 | 3rd Iteration: fix: Linux CI build fixes (#2) |

## 9. N/A-UNUSED-FEATURE - фичи, которые мы не используем (18)

| ID | Дата | Предмет | Причина |
|---|---|---|---|
| [CL-8dca17b4d](https://github.com/fewtarius/CachyLLama/commit/8dca17b4dc64756dd178f211e224ebe19d5c21f9) | 2026-09-01 | server: fix latent stale-spec-draft, get_n_draft_max, and dedup deferred checkpoint | speculative не используем |
| [CL-ce0a66429](https://github.com/fewtarius/CachyLLama/commit/ce0a664294b79b185d3dbbfc1571159bb0982555) | 2026-09-01 | spec: respect dp.n_max in DFlash, EAGLE3, MTP draft paths | speculative/MTP не используем |
| [CL-16ee42c84](https://github.com/fewtarius/CachyLLama/commit/16ee42c848f73d037aed5b8617748fb48310caae) | 2026-09-01 | dflash: apply aux_norm before fc in K/V injection paths | DFlash не используем |
| [CL-376bcc14d](https://github.com/fewtarius/CachyLLama/commit/376bcc14d5b2fdbf09f3c7d888c64f63a619c2cb) | 2026-08-27 | qwen4exp: add MTP draft head support | MTP не используем (наша сборка MTP не включает) |
| [CL-388733fde](https://github.com/fewtarius/CachyLLama/commit/388733fde46af600d643fcb7e1dea8f53d9eba00) | 2026-08-15 | laguna : expose t_h_nextn for DFlash draft + fix inp_out_ids crop crash | Laguna + DFlash |
| [CL-3e601a797](https://github.com/fewtarius/CachyLLama/commit/3e601a79767afb2e44c68a41af2d8479695fd7b2) | 2026-08-09 | models : DFlash draft support for Laguna-S-2.1 (decoder_arch = "laguna") | Laguna + DFlash |
| [CL-85a0c11e1](https://github.com/fewtarius/CachyLLama/commit/85a0c11e12fb17eb3329b0636b3f87bb1ff85cb7) | 2026-06-28 | Asad Ali Bhatti: fix(ssd): save/restore MTP ctx_dft and pending_h across cold-start restarts | MTP + SSD |
| [CL-37e857a3d](https://github.com/fewtarius/CachyLLama/commit/37e857a3d2089c0e54be64edf213901eea0c4668) | 2026-09-01 | server: skip conv_hash boundary purge for stateless embedding tasks | embedding не используем |
| [CL-d7d027d8c](https://github.com/fewtarius/CachyLLama/commit/d7d027d8ca41a31bd1763313bb07ac490fba42e7) | 2026-07-27 | fix(server): don't conflate mtmd capability with slot media content (issue #11) | vision/mtmd не используем |
| [CL-14261590e](https://github.com/fewtarius/CachyLLama/commit/14261590e99b30d1418a6c731b6f532f66580e64) | 2026-08-30 | chat: salvage tool calls and reasoning from raw text when PEG parse fails | robustness для DSML/Laguna-разметки, не для нашей модели |
| [CL-c0d864373](https://github.com/fewtarius/CachyLLama/commit/c0d8643738386aa691dc8d252d6f937ab7a7e8e0) | 2026-08-30 | chat: don't kill session when peg parse fails on complete output | то же |
| [CL-7043bf197](https://github.com/fewtarius/CachyLLama/commit/7043bf197eabf97215f06b33564b71d6150293a2) | 2026-06-01 | server : thread user_id field from request body to server_task | multi-user не используем |
| [CL-cfbdfb399](https://github.com/fewtarius/CachyLLama/commit/cfbdfb3996b94174226a99a324d6109d66b6130e) | 2026-06-01 | server : per-user concurrency cap and slot affinity | multi-user |
| [CL-6b7207d0b](https://github.com/fewtarius/CachyLLama/commit/6b7207d0b7640f7e67c8793ecba8526e33c7caac) | 2026-06-01 | server : page manager routes user_id to u/ namespace, no cross-user lookup | multi-user |
| [CL-7bdf8c253](https://github.com/fewtarius/CachyLLama/commit/7bdf8c253df8b05413d3b2d609ea733e2c50dc3c) | 2026-06-01 | server : return HTTP 429 when per-user concurrency cap is hit | multi-user |
| [CL-8b9fa2e68](https://github.com/fewtarius/CachyLLama/commit/8b9fa2e68b28c2dac9a67ebeb86bfd1d563f36e0) | 2026-06-01 | server : add --max-concurrent-per-user CLI flag and document the design | multi-user |
| [CL-ae7101da1](https://github.com/fewtarius/CachyLLama/commit/ae7101da14454e8556becaadf628dfaf84cd9b0d) | 2026-06-01 | common : add namespace_prefix to kv_ssd_init for u/ isolation | multi-user |
| [CL-a3ad6490b](https://github.com/fewtarius/CachyLLama/commit/a3ad6490b8afb396e23e465543ff392740eeabd1) | 2026-08-25 | fix(server): preserve user_id_ across task boundaries + prevent false-positive boundary on LCP match | multi-user |

## 10. N/A-FORK-INTERNAL - внутреннее дела форка (16)

| ID | Дата | Предмет | Причина |
|---|---|---|---|
| [CL-7b9afec66](https://github.com/fewtarius/CachyLLama/commit/7b9afec660efb3c485413c65d41c3cfa356df3df) | 2026-07-04 | 3rd Iteration: ci(vulkan): add Windows x64 Vulkan build with artifacts (#1) | CI форка (Vulkan) |
| [CL-65a2489c6](https://github.com/fewtarius/CachyLLama/commit/65a2489c62560b0c984a8d189f213ba25db5afbd) | 2026-08-15 | docs: refactor README.md and AGENTS.md for mature performance-focused fork | ребрендинг/доки |
| [CL-55935cfcb](https://github.com/fewtarius/CachyLLama/commit/55935cfcb965af3c85b511cf173a21391428c5b9) | 2026-06-14 | docs: rebrand README as CachyLLama | ребрендинг |
| [CL-148396174](https://github.com/fewtarius/CachyLLama/commit/1483961740e8792f1f12e3252caf1b512a6469b7) | 2026-08-01 | docs(agents): update RDNA3.5 patch status after upstream merge | доки форка |
| [CL-2480d170c](https://github.com/fewtarius/CachyLLama/commit/2480d170c7153aa48dec056d2dd542313ad11c1c) | 2026-07-22 | docs(agents): refresh CachyLLama-only patch table after upstream pull | доки форка |
| [CL-adfe31ecf](https://github.com/fewtarius/CachyLLama/commit/adfe31ecf23a152979dc570eebc42b7e771fead5) | 2026-08-29 | docs+scripts: memory/storage guarantees, post-memory-change bench profile | доки/скрипты форка |
| [CL-87f485eeb](https://github.com/fewtarius/CachyLLama/commit/87f485eebd09c1cda03d160e1dd363d3a60fb93a) | 2026-07-24 | fix(cuda): mark mmq_get_sum_size __host__ __device__ for nvcc device-context calls | фикс цепочки компиляции fork-only кода (mmq_get_sum_size в нашем дереве нет) |
| [CL-6bd9beb17](https://github.com/fewtarius/CachyLLama/commit/6bd9beb1750039574130ca292357d2b583c5d5d6) | 2026-07-23 | cuda : make ggml_cuda_get_physical_warp_size host-callable | то же: нужен только из-за fork-only mmq_get_sum_size |
| [CL-5044107be](https://github.com/fewtarius/CachyLLama/commit/5044107becc80af940386cf6f5234abd4955816e) | 2026-06-01 | detect: add CPU ISA auto-detection for cmake build flags | bash + /proc/cpuinfo; мы флаги ISA задаём явно (см. 06) |
| [CL-c761db7d9](https://github.com/fewtarius/CachyLLama/commit/c761db7d9de70cd391c98d06ec3ca0890566ad1c) | 2026-08-12 | fix(merge): repair references lost in upstream slot-restore refactor | merge-фикс форка |
| [CL-90cbb2e7e](https://github.com/fewtarius/CachyLLama/commit/90cbb2e7e8d27f45f8db7e2b70c5f4c5d678f90b) | 2026-08-24 | tools: re-add batched-bench to server-focused build | профиль сборки форка (server-only trim) |
| [CL-b9ed083e4](https://github.com/fewtarius/CachyLLama/commit/b9ed083e41b5ae79d942e6d0e66bd1f2e7b30429) | 2026-08-21 | build: re-enable llama-cli tool | то же |
| [CL-4d8fae05b](https://github.com/fewtarius/CachyLLama/commit/4d8fae05b4485c8b86646ee971ae68b5b8420912) | 2026-08-29 | server: define LLAMA_BUILD_UI compile flag when UI build is enabled | WebUI форка |
| [CL-4b1c2b659](https://github.com/fewtarius/CachyLLama/commit/4b1c2b659ad0ebe51676e5a685d681759326a0b0) | 2026-06-07 | ui : relax engine-strict for Node 23 compatibility | WebUI форка |
| [CL-4ec44dc10](https://github.com/fewtarius/CachyLLama/commit/4ec44dc10601165ac7427b5fa1c83bfe87d8bbf6) | 2026-08-30 | qa: fix smells and duplication during a QA review | внутренний code-QA |
| [CL-dd3fccf1a](https://github.com/fewtarius/CachyLLama/commit/dd3fccf1a95b3f91c140ca9859c9f94f1c402ca4) | 2026-08-12 | llama: keep DeepSeek lightning-indexer key cache f16 under quantized -ctk (nathanw1014) | DSA/Vulkan fused-ядра (не qwen4exp); как CONFIG-подтверждение см. 02, Приоритет 9 |

## 11. N/A-ISWA-PATH - правит класс памяти, не используемый нашим деревом (1)

| ID | Дата | Предмет | Комментарий |
|---|---|---|---|
| [CL-f629077d1](https://github.com/fewtarius/CachyLLama/commit/f629077d1337a4ed8b502ce7a42010cdcbb77705) | 2026-08-22 | server: keep prompt cache current + re-enable ISWA chunk reuse | qwen4exp использует llama_memory_hybrid_idx, ISWA не инстанцируется ([llama-model.cpp:2509-2529](src/llama-model.cpp:2509)); proof в [04-not-applicable.md](04-not-applicable.md), раздел 8. Перенесено из BLOCKED 2026-09-04 |
