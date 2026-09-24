# Не применимо - сгруппировано по причине

85 коммитов из 144. Причинно-следственные группы; ID кликабельны на коммиты.

## 1. Нет Vulkan-бэкенда (26)

Сборка CUDA/Windows; все Vulkan-ядра, шейдеры, тюнинг и их доки мимо.
Отдельно отмечено: CL-a17e0a9e1 - это поддержка gather-пути QSA на Vulkan
для CL-a657590ae; на CUDA тот же эффект даёт сам портированный gather (см.
02, Приоритет 1), Vulkan-обвязка не нужна.

| ID | Предмет |
|---|---|
| [CL-a17e0a9e1](https://github.com/fewtarius/CachyLLama/commit/a17e0a9e136bade8b483ac7d58513989769c59a0) | vulkan: widen topk_f32 pipelines (11 -> 14) for QSA indexer gather |
| [CL-c22b79e17](https://github.com/fewtarius/CachyLLama/commit/c22b79e174d4318b43896c68634f9454ef8576c6) | fix(merge): shader gen duplicate после merge upstream |
| [CL-887d408ba](https://github.com/fewtarius/CachyLLama/commit/887d408ba905b54d3db6f9f35857e3647055a5e7) | vulkan: coopmat lightning indexer dispatch |
| [CL-402d5a7e7](https://github.com/fewtarius/CachyLLama/commit/402d5a7e7bf14aeab97d2b9f002306cbaa0f72fe) | vulkan: fix Lightning Indexer + Strix Halo prefill/MoE perf |
| [CL-85d281a23](https://github.com/fewtarius/CachyLLama/commit/85d281a23695c8ea22e7d5f9c18be888282e5e61) | vulkan: C2 DSV4 sparse FA + coopmat indexer shaders |
| [CL-5f12949af](https://github.com/fewtarius/CachyLLama/commit/5f12949af4a323a88273346d40d2d961c5da1d7d) | vulkan: CONCAT_TRANSPOSE dispatch |
| [CL-5aed1f989](https://github.com/fewtarius/CachyLLama/commit/5aed1f989ba73f65313c5327382ab3d3ffb425c2) | vulkan: mmid row-list prepass |
| [CL-9902e67a0](https://github.com/fewtarius/CachyLLama/commit/9902e67a0753d468bd1d9a87676cba4f9b4a6e64) | vulkan: 32-wide subgroup for coopmat1 FA |
| [CL-e9691599a](https://github.com/fewtarius/CachyLLama/commit/e9691599a95ad184da3797e5db1b2fe11bf3cbd1) | vulkan: Psh query-major для GEMM2 |
| [CL-481c195a8](https://github.com/fewtarius/CachyLLama/commit/481c195a85ea6732ddd3788512b99bcfc1fed01c) | vulkan: hoist P-fragment load |
| [CL-33cc3c520](https://github.com/fewtarius/CachyLLama/commit/33cc3c520a54c91c0607bcb6c46e9a1e19554c70) | vulkan: FA dequant-once q4_0/q4_1/q5_0/q5_1 |
| [CL-404732f8c](https://github.com/fewtarius/CachyLLama/commit/404732f8c094c0a174f4439e549004f675247646) | vulkan: contiguize strided f16 KV |
| [CL-e36a9872a](https://github.com/fewtarius/CachyLLama/commit/e36a9872a6b8caf22c67567253f57d14fb027865) | vulkan: ggml_vk_fa_kv_native() |
| [CL-b3a9e851e](https://github.com/fewtarius/CachyLLama/commit/b3a9e851e1971d6ac40e713da7640990033f9470) | vulkan: concat_transpose shader |
| [CL-ae5458776](https://github.com/fewtarius/CachyLLama/commit/ae5458776ad4af3808fe9193d2b3573c3f5f3381) | vulkan: FA MMQ dot в fp32 before narrowing |
| [CL-0a506ca1c](https://github.com/fewtarius/CachyLLama/commit/0a506ca1caa920a2a0cd72e82f8b245859b259a0) | vulkan: bound command buffers by memory traffic |
| [CL-3cb0dc761](https://github.com/fewtarius/CachyLLama/commit/3cb0dc761993cc25669e9c4d3730d3c6f91f10dd) | vulkan: flush compute ctx before perf logger |
| [CL-d84ea1ca2](https://github.com/fewtarius/CachyLLama/commit/d84ea1ca2efe184bb1d923fe3c7ad681e778e66c) | vulkan: BF16 K-cache в Lightning Indexer |
| [CL-760e06e63](https://github.com/fewtarius/CachyLLama/commit/760e06e63f8169fd53c18bdd6581ad8f9e572da4) | vulkan: DSV4 Lightning Indexer fused op |
| [CL-1b7ea8149](https://github.com/fewtarius/CachyLLama/commit/1b7ea814934a8cf6a8b21968f7062fa101ca91ab) | vulkan: DSV4 hyper-connection fused ops |
| [CL-a16163278](https://github.com/fewtarius/CachyLLama/commit/a1616327810895158b1f291eb053b52269391886) | vulkan: tiled transpose для 0<->2 CONT |
| [CL-c0407cd21](https://github.com/fewtarius/CachyLLama/commit/c0407cd219454e3f27f460bbef3e85ebe5593d8e) | vulkan: FA dequant+transpose safety gates |
| [CL-bd4f2875b](https://github.com/fewtarius/CachyLLama/commit/bd4f2875b177abd3ada0bdd39318293cdeb3b4d5) | vulkan: dequant q8_0 KV once in coopmat1 |
| [CL-1c19480da](https://github.com/fewtarius/CachyLLama/commit/1c19480dafb6d705613d5492bdeaed015c8a89d9) | fix(vulkan): auto-lower nodes_per_submit for APU/iGPU |
| [CL-15e399d01](https://github.com/fewtarius/CachyLLama/commit/15e399d0176edae69d884133ffcce55e5766057b) | docs(vulkan): CEZANNE_NOTES.md |
| [CL-0da40ed6c](https://github.com/fewtarius/CachyLLama/commit/0da40ed6cd40e787848dc1333ba9711a821a2fc6) | docs(vulkan): RDNA3 (Phoenix) tuning notes |

## 2. ROCm / HIP / RDNA3.5 (gfx1151) (4)

CL-2b0c4dcfd формально правит ggml-cuda, но крах (HSA memory fault через
MMA write_back) возможен только на AMD-сборке ROCm; на NVIDIA dp4a-размер
`sum[]` корректен. CL-71d1e8f2f правит MMQ-tile только для cfg rdna3_5.

| ID | Предмет |
|---|---|
| [CL-5ff23cdf3](https://github.com/fewtarius/CachyLLama/commit/5ff23cdf339d58cf8e1a0a1f629dd9151b53c71d) | perf(rocm): RDNA3.5 Strix Halo tuning |
| [CL-71d1e8f2f](https://github.com/fewtarius/CachyLLama/commit/71d1e8f2fbf769e2b61d50742188794f4acf3303) | fix(cuda): bump rdna3_5 MMQ I 48 -> 64 |
| [CL-2b0c4dcfd](https://github.com/fewtarius/CachyLLama/commit/2b0c4dcfd68caeda50c89538cba57b438c9b08dc) | fix(ggml-cuda): size MMQ sum[] for dp4a and MMA |
| [CL-d95602b56](https://github.com/fewtarius/CachyLLama/commit/d95602b56f0bc93b01094122110ee0ce0c0b1729) | ci(self-hosted): gfx1151 HIP smoke build |

## 3. macOS / Metal (2)

| ID | Предмет |
|---|---|
| [CL-b782fb874](https://github.com/fewtarius/CachyLLama/commit/b782fb8749a318ca1c154a6e0e918923bcb6a171) | build(macos): fix CachyLLama build on macOS |
| [CL-b6bf93092](https://github.com/fewtarius/CachyLLama/commit/b6bf93092c947bbf7fd43793eb6cb2deed7f27ae) | fix(ssd): detect host RAM on macOS (sysctl) |

## 4. Чужие архитектуры моделей / UMA-специфика (4)

DSML/Laguna парсеры - разметка DeepSeek-V4-Flash и Laguna, наша модель их не
производит. CL-7140f5fae - auto-size SSD-бюджета для UMA APU (модель вычитается
из RAM-бюджета); у нас дискретная видеопамять, подсистемы SSD тоже нет.

| ID | Предмет |
|---|---|
| [CL-7140f5fae](https://github.com/fewtarius/CachyLLama/commit/7140f5fae1d38e4aa1e8c773471830b171f14a02) | ssd: subtract model footprint from auto-size budget on UMA APUs |
| [CL-bc499b060](https://github.com/fewtarius/CachyLLama/commit/bc499b060c2ceb55a9b426fb0b0e0bf41a90d59a) | chat: DSML/PEG parser for DeepSeek-V4-Flash tool calls |
| [CL-7caa42de1](https://github.com/fewtarius/CachyLLama/commit/7caa42de1fd10bcd8f9f6bdb72b0b396102bd1d6) | chat: strip unclosed/orphaned DSML tokens |
| [CL-af6fbcb2a](https://github.com/fewtarius/CachyLLama/commit/af6fbcb2a4e97771da66de1f7e06eb2023a17a97) | server: Laguna tag parser greedily consuming arg_key |

## 5. Linux-only API (madvise / mincore / /proc, Linux CI) (15)

Всё семейство MoE expert residency построено на madvise(MADV_FREE/COLD/WILLNEED)
по MAP_SHARED mmap модели + mincore() + /proc - на Windows нет аналогов.
moe tracking API и residency docs - часть того же. a2b13f5ab - CI-фиксы Linux.
Вместо этого у нас работает стандартный mmgr + Page Cache Windows.

| ID | Предмет |
|---|---|
| [CL-3ebbc7ea4](https://github.com/fewtarius/CachyLLama/commit/3ebbc7ea43b4e7284e24a1b90d51bbee05d36f9d) | feat(moe-offload): Phase 1 - per-token expert selection + madvise |
| [CL-2c8642383](https://github.com/fewtarius/CachyLLama/commit/2c86423831df3eeabd23e51b54bb93d8663c2659) | feat(moe-offload): Phase 2 - R+F scoring + co-activation |
| [CL-8edfad218](https://github.com/fewtarius/CachyLLama/commit/8edfad21805c22c8131eaec70456e548d81178db) | fix(moe-residency): MADV_FREE + larger cache |
| [CL-f6c412a61](https://github.com/fewtarius/CachyLLama/commit/f6c412a612b986dcc9107e126f55de07323d9d79) | moe-residency: MADV_FREE on MAP_SHARED + observability (mincore) |
| [CL-880d52c07](https://github.com/fewtarius/CachyLLama/commit/880d52c072a0afc6ccbb65cdafab3c08112fd89b) | fix(moe-offload): LLAMA_LOG_WARN for residency |
| [CL-bc00778cd](https://github.com/fewtarius/CachyLLama/commit/bc00778cdfda54433485dcb9dfc0e2822fd7ca4a) | perf(moe-offload): bump defaults |
| [CL-aa4644dc6](https://github.com/fewtarius/CachyLLama/commit/aa4644dc66bc9a5d9a25539f28691ec8e2e2632a) | fix(moe-offload): top-K from F32 probs |
| [CL-ebff3e87c](https://github.com/fewtarius/CachyLLama/commit/ebff3e87c3dedf0833991259f1a9871507561c71) | feat(moe-offload): argsort + topk tensors |
| [CL-668dc5150](https://github.com/fewtarius/CachyLLama/commit/668dc5150820c2cce1500867fbb4687f8d367c7c) | fix(moe-offload): synchronize before argsort read |
| [CL-b40698c7f](https://github.com/fewtarius/CachyLLama/commit/b40698c7f64f070ef84176a5d925f80ddeb2f12c) | fix(merge): --load-mode + residency check |
| [CL-34106161b](https://github.com/fewtarius/CachyLLama/commit/34106161be4b6c0e9f0c18a64ea5c694f6177eb8) | docs: MoE expert residency subsystem |
| [CL-b69de8497](https://github.com/fewtarius/CachyLLama/commit/b69de84971815ba49a7dc57b062633f8c4369c9b) | feat(server): MoE expert activation tracking API |
| [CL-1ffbac42a](https://github.com/fewtarius/CachyLLama/commit/1ffbac42a424aa32065a628d4ad548b7443ffef9) | feat: MoE expert activation tracking API (Phase 1) |
| [CL-86855c4ae](https://github.com/fewtarius/CachyLLama/commit/86855c4ae9cf395c05b211aac9dee9ca83e5a688) | fix(moe-offload): skip empty routing tensors |
| [CL-a2b13f5ab](https://github.com/fewtarius/CachyLLama/commit/a2b13f5ab98766a168ec5f4a97ad5f37a82c5c66) | fix: Linux CI build fixes (#2) |

## 6. Фичи, которые мы не используем (18)

Speculative decoding (DFlash/EAGLE3/MTP) - не включаем; Laguna/mtmd/embedding -
не наши модели/режимы; multi-user (user_id namespace, per-user caps) - сервер
однопользовательский; PEG-salvage правки - про DSML/Laguna-разметку.

| ID | Предмет | Причина |
|---|---|---|
| [CL-8dca17b4d](https://github.com/fewtarius/CachyLLama/commit/8dca17b4dc64756dd178f211e224ebe19d5c21f9) | stale-spec-draft, get_n_draft_max | spec |
| [CL-ce0a66429](https://github.com/fewtarius/CachyLLama/commit/ce0a664294b79b185d3dbbfc1571159bb0982555) | dp.n_max in DFlash/EAGLE3/MTP | spec |
| [CL-16ee42c84](https://github.com/fewtarius/CachyLLama/commit/16ee42c848f73d037aed5b8617748fb48310caae) | dflash aux_norm before fc | DFlash |
| [CL-376bcc14d](https://github.com/fewtarius/CachyLLama/commit/376bcc14d5b2fdbf09f3c7d888c64f63a619c2cb) | qwen4exp: MTP draft head | MTP не включён |
| [CL-388733fde](https://github.com/fewtarius/CachyLLama/commit/388733fde46af600d643fcb7e1dea8f53d9eba00) | laguna t_h_nextn + inp_out_ids crop | Laguna+DFlash |
| [CL-3e601a797](https://github.com/fewtarius/CachyLLama/commit/3e601a79767afb2e44c68a41af2d8479695fd7b2) | DFlash для Laguna-S-2.1 | Laguna+DFlash |
| [CL-85a0c11e1](https://github.com/fewtarius/CachyLLama/commit/85a0c11e12fb17eb3329b0636b3f87bb1ff85cb7) | SSD save/restore MTP ctx_dft | MTP+SSD |
| [CL-37e857a3d](https://github.com/fewtarius/CachyLLama/commit/37e857a3d2089c0e54be64edf213901eea0c4668) | conv_hash purge для embedding | embedding |
| [CL-d7d027d8c](https://github.com/fewtarius/CachyLLama/commit/d7d027d8ca41a31bd1763313bb07ac490fba42e7) | mtmd capability vs slot media | vision/mtmd |
| [CL-14261590e](https://github.com/fewtarius/CachyLLama/commit/14261590e99b30d1418a6c731b6f532f66580e64) | salvage tool calls после PEG fail | чужая разметка |
| [CL-c0d864373](https://github.com/fewtarius/CachyLLama/commit/c0d8643738386aa691dc8d252d6f937ab7a7e8e0) | not kill session on peg fail | чужая разметка |
| [CL-7043bf197](https://github.com/fewtarius/CachyLLama/commit/7043bf197eabf97215f06b33564b71d6150293a2) | user_id threading | multi-user |
| [CL-cfbdfb399](https://github.com/fewtarius/CachyLLama/commit/cfbdfb3996b94174226a99a324d6109d66b6130e) | per-user cap + slot affinity | multi-user |
| [CL-6b7207d0b](https://github.com/fewtarius/CachyLLama/commit/6b7207d0b7640f7e67c8793ecba8526e33c7caac) | u/ namespace routing | multi-user |
| [CL-7bdf8c253](https://github.com/fewtarius/CachyLLama/commit/7bdf8c253df8b05413d3b2d609ea733e2c50dc3c) | HTTP 429 per-user cap | multi-user |
| [CL-8b9fa2e68](https://github.com/fewtarius/CachyLLama/commit/8b9fa2e68b28c2dac9a67ebeb86bfd1d563f36e0) | --max-concurrent-per-user | multi-user |
| [CL-ae7101da1](https://github.com/fewtarius/CachyLLama/commit/ae7101da14454e8556becaadf628dfaf84cd9b0d) | namespace_prefix в kv_ssd_init | multi-user |
| [CL-a3ad6490b](https://github.com/fewtarius/CachyLLama/commit/a3ad6490b8afb396e23e465543ff392740eeabd1) | preserve user_id_ across boundaries | multi-user |

## 7. Внутренние дела форка (16)

Merge-фиксы после upstream-pull, ребрендинг/доки/AGENTS, WebUI форка,
server-focused trim сборки, CI, bash-скрипты. Отдельно: CL-87f485eeb и
CL-6bd9beb17 - починка цепочки компиляции `__host__/__device__` вокруг
fork-only `mmq_get_sum_size`; в нашем дереве этого символа нет
([common.cuh:374](ggml/src/ggml-cuda/common.cuh:374) по-прежнему
`__device__`-only и этого достаточно), переносить нечего.
CL-dd3fccf1a - DeepSeek DSA + Vulkan fused-ядра; как config-правило учтено
в 02, Приоритет 9 (у нас ctk/ctv=f16).

| ID | Предмет |
|---|---|
| [CL-7b9afec66](https://github.com/fewtarius/CachyLLama/commit/7b9afec660efb3c485413c65d41c3cfa356df3df) | ci(vulkan): Windows x64 Vulkan build (#1) |
| [CL-65a2489c6](https://github.com/fewtarius/CachyLLama/commit/65a2489c62560b0c984a8d189f213ba25db5afbd) | docs: refactor README/AGENTS |
| [CL-55935cfcb](https://github.com/fewtarius/CachyLLama/commit/55935cfcb965af3c85b511cf173a21391428c5b9) | docs: rebrand README |
| [CL-148396174](https://github.com/fewtarius/CachyLLama/commit/1483961740e8792f1f12e3252caf1b512a6469b7) | docs(agents): RDNA3.5 patch status |
| [CL-2480d170c](https://github.com/fewtarius/CachyLLama/commit/2480d170c7153aa48dec056d2dd542313ad11c1c) | docs(agents): CachyLLama-only patch table |
| [CL-adfe31ecf](https://github.com/fewtarius/CachyLLama/commit/adfe31ecf23a152979dc570eebc42b7e771fead5) | docs+scripts: memory guarantees, bench profile |
| [CL-87f485eeb](https://github.com/fewtarius/CachyLLama/commit/87f485eebd09c1cda03d160e1dd363d3a60fb93a) | fix(cuda): mmq_get_sum_size __host__ __device__ |
| [CL-6bd9beb17](https://github.com/fewtarius/CachyLLama/commit/6bd9beb1750039574130ca292357d2b583c5d5d6) | cuda: warp_size host-callable |
| [CL-5044107be](https://github.com/fewtarius/CachyLLama/commit/5044107becc80af940386cf6f5234abd4955816e) | detect: CPU ISA auto-detection (bash+/proc) |
| [CL-c761db7d9](https://github.com/fewtarius/CachyLLama/commit/c761db7d9de70cd391c98d06ec3ca0890566ad1c) | fix(merge): references lost in slot-restore refactor |
| [CL-90cbb2e7e](https://github.com/fewtarius/CachyLLama/commit/90cbb2e7e8d27f45f8db7e2b70c5f4c5d678f90b) | tools: re-add batched-bench |
| [CL-b9ed083e4](https://github.com/fewtarius/CachyLLama/commit/b9ed083e41b5ae79d942e6d0e66bd1f2e7b30429) | build: re-enable llama-cli |
| [CL-4d8fae05b](https://github.com/fewtarius/CachyLLama/commit/4d8fae05b4485c8b86646ee971ae68b5b8420912) | server: LLAMA_BUILD_UI flag |
| [CL-4b1c2b659](https://github.com/fewtarius/CachyLLama/commit/4b1c2b659ad0ebe51676e5a685d681759326a0b0) | ui: relax engine-strict Node 23 |
| [CL-4ec44dc10](https://github.com/fewtarius/CachyLLama/commit/4ec44dc10601165ac7427b5fa1c83bfe87d8bbf6) | qa: fix smells and duplication |
| [CL-dd3fccf1a](https://github.com/fewtarius/CachyLLama/commit/dd3fccf1a95b3f91c140ca9859c9f94f1c402ca4) | llama: DeepSeek indexer key cache f16 |

## 8. ISWA-путь: класс памяти не используется нашим деревом (1)

| ID | Предмет |
|---|---|
| [CL-f629077d1](https://github.com/fewtarius/CachyLLama/commit/f629077d1337a4ed8b502ce7a42010cdcbb77705) | server: keep prompt cache current + re-enable ISWA chunk reuse |

Перенесено из BLOCKED 2026-09-04 после проверки (TASK D). Доказательство N/A по
нашему дереву:

- qwen4exp создаёт `llama_memory_hybrid_idx`
  ([llama-model.cpp:2509-2529](src/llama-model.cpp:2509)), аргумент unified =
  `cparams.kv_unified` ([:2526](src/llama-model.cpp:2526)); needs_mem_idx для
  qwen4exp ([:2462](src/llama-model.cpp:2462)). SWA-ветка
  ([llama-model.cpp:2489-2508](src/llama-model.cpp:2489)) для qwen4exp не
  проходит: в [src/models/qwen4exp.cpp](src/models/qwen4exp.cpp) нет SWA-ключей.
- Внутри hybrid обычный `llama_kv_cache` + `llama_memory_recurrent`
  ([llama-memory-hybrid.h:89-90](src/llama-memory-hybrid.h:89)). Класс
  `llama_kv_cache_iswa` для qwen4exp не инстанцируется никогда.
- Коммит правит `llama_kv_cache_iswa::get_can_shift()`
  ([llama-kv-cache-iswa.cpp:253-257](src/llama-kv-cache-iswa.cpp:253)) - путь,
  недостижимый для нашей модели. Фактический gate K-shift для qwen4exp -
  `llama_kv_cache::get_can_shift()`
  ([llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189)), который и так
  возвращает false для mrope (n_pos_per_embd=4,
  [qwen4exp.cpp:521](src/models/qwen4exp.cpp:521)); ctx_shift тогда
  отключается в [common/common.cpp:1457](common/common.cpp:1457). Это же
  зафиксировано в UPSTREAM-строке CL-4db9548a8.

Server/task-часть коммита (keep prompt cache current) привязана к fork-only
prompt-cache подсистеме - без базы переносить некуда.
