# Реестр проверенных оптимизаций из fewtarius/CachyLLama

См. также: [Сводный отчёт по обоим форкам](../SUMMARY.md)

Форк ориентирован на AMD APU (Strix Halo / gfx1151): Vulkan, ROCm, Linux.
Это принципиально отличается от сетапа пользователя (Windows + CUDA + RTX 3080),
поэтому значительная часть форка не применима. При этом в форке есть
аппаратно-независимые фиксы (QSA, sched, recurrent memory), которые бьют
точно в нашу модель qwen4exp.

Данные: форк HEAD `8dca17b4dc64756dd178f211e224ebe19d5c21f9` (2026-09-01),
merge-base с upstream `3466812d1`, форк ahead 165 (144 non-merge коммита),
behind 40. Источник истины по списку: `D:\dev\llama_compiling\_forks_opt\cl_full_log.txt`,
полные сообщения ключевых коммитов: `D:\dev\llama_compiling\_forks_opt\cl_msgs_sel.txt`,
сырой инвентарь: [07-raw-inventory.md](07-raw-inventory.md).

Правка 2026-09-04: CL-f629077d1 (ISWA chunk reuse) переклассифицирован из
BLOCKED в N/A-ISWA-PATH (qwen4exp использует llama_memory_hybrid_idx, ISWA для
него не инстанцируется - см. 04-not-applicable.md, раздел 8). создан
05-blocked-or-needs-diff.md с детализацией BLOCKED-групп и объёмом SSD-базы.

Правка 2026-09-11: SSD-база перенесена в дерево, ветка `ssd-kv-cache`
(CL-6a0db500c base + CL-56dca0825 + CL-cc60f8912, net-diff с squash'ом).
POSIX-адаптация под MSVC: mmap -> _open_osfhandle/MapViewOfFile, madvise ->
no-op, host-RAM через GlobalMemoryStatusEx, mkdir -> _mkdir, unistd ->
process.h. Флаги `--cache-ssd*` видны в `llama-server --help`. Unit-тест
`test-kv-ssd-user-isolation` проходит на Windows. Фиксы группы SSD-cache
разблокированы, но ещё не переносились; seq_rm_attn_only-фиксы остаются
BLOCKED (API в дереве нет).

## Целевой сетап

- CPU: i7-12700K (E-ядра отключены, AVX-512 разблокирован, нет AVX512-BF16)
- GPU: RTX 3080 10GB (sm_86), CUDA backend
- RAM: 64GB DDR4-3600
- ОС: Windows 11 25H2, MSVC + Ninja Release
- Сборка: GGML_CUDA=ON, CUDA arch 86, GGML_NATIVE/LTO/CCACHE/OPENMP,
  AVX2/AVX512/VBMI/VNNI/AVX_VNNI/BMI2, GGML_CUDA_GRAPHS=ON, FA=ON + FA_ALL_QUANTS
- НЕ используется: Vulkan, ROCm, Metal, RPC
- Модель: Qwen3.8-Flash-Next GGUF unsloth UD-Q3_K_XL, архитектура `qwen4exp`
  (QSA/Lightning Indexer через ggml_top_k, gated delta-net + conv,
  merged gate_up_exps через mul_mat_id, per_layer_token_embd PLE)
- Запуск: ctx 90000, ctk=f16 ctv=f16, fa=auto, kvu=true, parallel=1,
  b/ub=1024, t=13 tb=16, cache-ram 3000, ctx-checkpoints 3, lazy-mode=on,
  load-mode=mlock, fit=off, jinja=on, ot="(.ffn_.*exp|per_layer_token_embd)=CPU",
  timeout 1800, lv=4, top-k 20 / min-p 0
- НЕ используется: speculative/MTP, vision (mtmd), LoRA, grammar, embedding, TTS
- Дерево пользователя: `D:/dev/llama_compiling/llama.cpp`, HEAD `413cfde3b`,
  merge-base с upstream `8887a48f0`, ahead 24 (собственные патчи: qwen4exp
  QSA top_k-skip, IQP AVX-512/VNNI, FA mma f16 ncols2 head256/gqa12,
  small F32 GEMM, fast tensor names, fused rms_norm, AVX-VNNI MSVC detect,
  NUMA fallback)

## Структура каталога

| Файл | Содержание |
|---|---|
| [README.md](README.md) | Это: сетап, легенда, счётчики, ограничения |
| [01-registry.md](01-registry.md) | Все 144 коммита форка, статус каждого |
| [02-applicable.md](02-applicable.md) | Шортлист применимого: что править, где, как |
| [03-already-upstream.md](03-already-upstream.md) | Эквивалент уже есть в upstream/нашем дереве |
| [04-not-applicable.md](04-not-applicable.md) | N/A, сгруппировано по причине |
| [05-blocked-or-needs-diff.md](05-blocked-or-needs-diff.md) | Детализация BLOCKED-групп: SSD-база (объём diff, POSIX-блокировки), seq_rm_attn_only, ISWA |
| [06-build-flags-audit.md](06-build-flags-audit.md) | Аудит флагов сборки: что форк меняет/не меняет |
| [07-raw-inventory.md](07-raw-inventory.md) | Сырой инвентарь (списки коммитов, площади) |

## Легенда статусов

| Статус | Значение |
|---|---|
| CUDA-APPLICABLE | Код применим в нашем дереве (CUDA/Windows), нужно портировать |
| CONFIG-ONLY | Код уже есть, нужен только параметр запуска (встречается как пометка внутри элементов) |
| PORTED | Перенесено в наше дерево (2026-09-11, ветка `ssd-kv-cache`) |
| UPSTREAM | Эквивалент уже в upstream, переносить нечего |
| BLOCKED | Применимо только после портирования другой подсистемы форка (SSD-cache / checkpoint ring / seq_rm_attn_only) |
| N/A-VULKAN | Vulkan-ядра и их доки |
| N/A-ROCm | ROCm/HIP и RDNA3.5 (gfx1151)-специфика |
| N/A-METAL | macOS/Metal |
| N/A-ARCH | Чужие модели/парсеры шаблонов, UMA-специфика |
| N/A-LINUX-ONLY | madvise/mincore, /proc, Linux CI |
| N/A-UNUSED-FEATURE | Фичи, которые мы не используем (spec/MTP/DFlash, DSV4, multi-user, mtmd, embedding) |
| N/A-FORK-INTERNAL | Внутреннее дела форка: merge-фиксы, UI, ребрендинг, доки, CI (расширение легенды) |
| N/A-ISWA-PATH | Правит llama_kv_cache_iswa - класс памяти, который qwen4exp не инстанцирует (llama-model.cpp:2489-2529) |

## Счётчики (144 коммита)

| Статус | Кол-во |
|---|---|
| PORTED | 3 |
| CUDA-APPLICABLE | 6 |
| UPSTREAM | 1 |
| BLOCKED | 48 |
| N/A-VULKAN | 26 |
| N/A-ROCm | 4 |
| N/A-METAL | 2 |
| N/A-ARCH | 4 |
| N/A-LINUX-ONLY | 15 |
| N/A-UNUSED-FEATURE | 18 |
| N/A-FORK-INTERNAL | 16 |
| N/A-ISWA-PATH | 1 |
| **Итого** | **144** |

## Ключевые ограничения дерева (действуют на весь реестр)

1. Нет Vulkan/ROCm/Metal: все 26+4+2 коммита бэкендов вылетыают из скоупа
   автоматически; CUDA-пути они не затрагивают.
2. `qwen4exp` использует IMROPE, `n_pos_per_embd() == 4`
   ([llama-model.cpp:2960](src/llama-model.cpp:2960),
   [llama-hparams.cpp:260](src/llama-hparams.cpp:260)). K-shift уже отключён
   для таких моделей ([llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189)).
3. В нашем дереве НЕТ API `seq_rm_attn_only` - на нём держатся 3+ коммита
   форка, они в BLOCKED.
4. ~~В нашем дереве НЕТ подсистем форка: SSD KV cache~~ - SSD KV cache +
   sys-prompt cache перенесены 2026-09-11 (ветка `ssd-kv-cache`). Остальные
   подсистемы форка (deferred_create_final_checkpoint, conv_hash,
   moe-residency) по-прежнему отсутствуют. SSD-фиксы разблокированы и ждут
   переноса; seq_rm_attn_only-фиксы остаются BLOCKED. Детализация:
   [05-blocked-or-needs-diff.md](05-blocked-or-needs-diff.md).
5. Linux-only syscall-и (madvise MADV_COLD/MADV_FREE, mincore): всё семейство
   MoE expert residency неприменимо на Windows.
6. Числа эффекта в отчётах: если приведено число - оно из сообщения коммита
   (замер автора, обычно на Strix Halo / Vulkan, если не сказано иное).
   Без замера помечено "(оценка, не измерено)".
7. Внимательно с SHA: в этом форке не существует `4892d5791`, `e2234e824`,
   `a5889b2d8` и др. - они из чужих списков, сюда не ссылаться.
   PR-номера: #1=7b9afec66, #2=a2b13f5ab, #3=8b2cf6c66.
