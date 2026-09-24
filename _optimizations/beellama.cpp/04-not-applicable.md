# 04-not-applicable.md
# N/A: сгруппировано по причине

Индвидуальные строки реестра (см. [01-registry.md](01-registry.md)) + укрупнённые
группы по Layer B логам (`D:\dev\llama_compiling\bee_audit\log_*.txt`).

## N/A-TURBOQUANT (5 именованных + масса в логах)

Модель UD-Q3_K_XL, KV f16: TurboQuant/TCQ/Q2_0 не участвуют ни в PP, ни в TG.

- [BE-1010625c5](https://github.com/Anbeeld/beellama.cpp/commit/1010625c5),
  [BE-375555536](https://github.com/Anbeeld/beellama.cpp/commit/375555536)
  (спецпроверка e). Механизм (по диффам fattn.cu и сообщениям): prefill-путь
  flash-attention для turbo-квантованных K/V - bulk-dequant блока turbo-K/V в
  fp16 перед MMA, чтобы использовать tensor cores вместо скалярного dequant.
  Цитаты из сообщения автора: "pp4096 588->1113 tok/s (1.9x)", "1.78x (98.8%
  of q8_0)", стенд RTX 3090 + Qwen3.5-27B Q6_K. К f16 KV не относится.
- [BE-f68ad76b6](https://github.com/Anbeeld/beellama.cpp/commit/f68ad76b6) FWHT для turbo.
- [BE-1f06a2dfb](https://github.com/Anbeeld/beellama.cpp/commit/1f06a2dfb) Q2_0 ternary support
  (плюс парный d36a9bf7a) - новый квант-тип, не ускорение существующего Q3_K-пути.
- [BE-8e6ced0c3](https://github.com/Anbeeld/beellama.cpp/commit/8e6ced0c3),
  c99c23018 - turbo KV + partial offload.
- Внутри mega-preserve `1a285c4d1`: turbo-quant-cuda.cuh +1356, ggml-turbo-quant.c +843
  и пр. - сами пути TurboQuant.
- В Layer B логе CUDA (115 коммитов) почти всё остальное - KVarN/Turbo/TCQ
  fattn-kvarn*, turbo*, tcq* файлы (исключались из выборки по правилам Layer B;
  попали в `log_cuda_general.txt` только как контекст).

## N/A-KVARN-LOWBIT (3+)

Запуск: `--cache-type-k f16 --cache-type-v f16` + kvu=true.

- [BE-098e00d34](https://github.com/Anbeeld/beellama.cpp/commit/098e00d34) Generalize KVarN native
  decode and parallelize CUDA builds.
- [BE-2446c0ed4](https://github.com/Anbeeld/beellama.cpp/commit/2446c0ed4) q6_0 KV cache.
- [BE-83429d46a](https://github.com/Anbeeld/beellama.cpp/commit/83429d46a) KV-tail (KVarN-семейство).

## N/A-SPEC-DECODING (6 именованных + семейство DFlash/SD-* ~100+)

Спекуляцию не используем (--spec-draft-*, DFlash, MTP, self-spec - off).

- [BE-11f45ed34](https://github.com/Anbeeld/beellama.cpp/commit/11f45ed34) Fix graph number
  calculation: дельта целиком в `LLM_ARCH_DFLASH` ветке `graph_max_nodes`
  (`dflash_selector_rank > 0`, 12*n_tensors вместо 8*n_tensors + удалённый
  selector_tokens adд-on). Generic ветка для нашей модели не тронута.
- [BE-f5a7ec15d](https://github.com/Anbeeld/beellama.cpp/commit/f5a7ec15d) mrope fix (speculative
  + conversion + dflash).
- [BE-446320565](https://github.com/Anbeeld/beellama.cpp/commit/446320565) GPU argmax for DFlash
  drafter: eliminate 15.9MB transfer + CPU scan - оптимизация drafter'а.
- [BE-f38d36dbf](https://github.com/Anbeeld/beellama.cpp/commit/f38d36dbf) async tape GDN overlap
  (GDN overlap для spec-дерева).
- [BE-91357ddcc](https://github.com/Anbeeld/beellama.cpp/commit/91357ddcc) + merge
  [BE-07ac3cec6](https://github.com/Anbeeld/beellama.cpp/commit/07ac3cec6) (PR#25 chimpera):
  аrgmax topk kernel K<=32 -> K<=64 - под DFlash reduced verifier (Gemma4
  top_k=64). Наш сэмплер top-k=20, не задевает лимит; спецпроверка b.
- Семейства `SD-*` (SD-067/068/071: rejection sampling, DDTree KV, best-first heap),
  batched DFlash draft (ff0444e46), dflash selector/plumbing (e91bca536, 73d4d1267,
  cab1fb597 и др.) - всё спекулятивное.

## N/A-ROCm (1+)

ROCm/HIP/gfx не собираются.

- [BE-8b250fba2](https://github.com/Anbeeld/beellama.cpp/commit/8b250fba2) Fix HIP FA launch
  bounds (+ Windows CPU packaging: windows-часть - packaging релизов форка).
- Релизный/CI ROCm-массивы форка (группа).

## N/A-ARCH (1)

- [BE-0b035b3a2](https://github.com/Anbeeld/beellama.cpp/commit/0b035b3a2) ARM64 fix.

## N/A-UNUSED (3)

Корректность/fallback путей, которых нет в нашем графе/сборке.

- [BE-0ef8c42f4](https://github.com/Anbeeld/beellama.cpp/commit/0ef8c42f4) reject unsupported
  BF16 scale (у нас f16/f32; FF16-only путь).
- [BE-9f268fb03](https://github.com/Anbeeld/beellama.cpp/commit/9f268fb03) reject unsupported
  CPU FA types (FA работает на CUDA, CPU-FA не используется).
- [BE-76e3f43de](https://github.com/Anbeeld/beellama.cpp/commit/76e3f43de) cpu: support f16
  out_prod fallback (+91 строк ops.cpp, новый тест test-cpu-out-prod.cpp) -
  feature-add fallback; out_prod f16 на CPU у нас не вызывается.

## N/A-FORK-INTERNAL (10)

- [BE-1a285c4d1](https://github.com/Anbeeld/beellama.cpp/commit/1a285c4d1) preserve TurboQuant
  and TCQ backends - mega-коммит дивергенции (163 файла, +18279/-681). Содержит
  перенос чужих путей целиком (включая gated_delta_net.cu +558, ssm-conv.cu +128,
  argmax.cu 486, set-rows.cu +361, topk-moe.cu, metal turbo файлы). Отдельных
  generic-ускорений внутри не найдено; tree-GDN - кандидат на ручной разбор
  (см. 01-registry.md).
- [BE-16c76b14a](https://github.com/Anbeeld/beellama.cpp/commit/16c76b14a) +
  [BE-fc9ab2bd6](https://github.com/Anbeeld/beellama.cpp/commit/fc9ab2bd6) prompt-cache
  transactional subsystem (35+19 файлов, +3513/+785) - собственная подсистема
  форка, KVarN-центричная; в наше дерево не переносима точечно.
- [BE-c8da45a37](https://github.com/Anbeeld/beellama.cpp/commit/c8da45a37) Fix prompt-cache host
  budget enforcement (учёт активных checkpoints в cache-ram бюджете) - фикс той же
  подсистемы.
- [BE-13ad92aaa](https://github.com/Anbeeld/beellama.cpp/commit/13ad92aaa) stop deriving
  checkpoint budget from cache-ram - убирает derived-бюджет (cache_ram/2), который
  форк сам же ввёл; оставляет только max-3 copy cap. В НАШЕМ дереве вывода бюджета
  чекпойнтов из cache-ram нет (grep server-context.cpp: только n_ctx_checkpoints) ->
  чинить нечего.
- [BE-ac792986d](https://github.com/Anbeeld/beellama.cpp/commit/ac792986d) preserve fused GDN
  4D state fast path - compat со СТАРЫМ 4D state layout (см. спецпроверку g):
  ggml.c принимает и 3D (S_v*S_v*H, K, n_seqs), и 4D (S_v, S_v, H, n_seqs);
  ggml-cpu/ops.cpp детектит 4D; delta-net-base.cpp перестал респейпить. Наше
  дерево уже на чистом 3D API (`ggml.c:6316` и вызов со снапшот-слотами) - обратная
  совместимость не нужна.
- [BE-5eaba174c](https://github.com/Anbeeld/beellama.cpp/commit/5eaba174c) reduce default FA pair
  matrix - build-матрица FA под типы форка (наш FA_ALL_QUANTS-путь не меняет).
- [BE-df8933c26](https://github.com/Anbeeld/beellama.cpp/commit/df8933c26) Allow ignoring
  uncompiled CUDA FA pairs - plumbing сборки матрицы форка.
- [BE-a596a5c72](https://github.com/Anbeeld/beellama.cpp/commit/a596a5c72) HALF_QUANTS build mode
  - флаг сборки форка (см. [06-build-flags-audit.md](06-build-flags-audit.md)).
- [BE-94605e9fe](https://github.com/Anbeeld/beellama.cpp/commit/94605e9fe) ci: seed version caches
  - релизный CI форка.
- BE-5678fc3b8 restore D512 FA selector - удаление fork-override после upstream-слияния;
  к нашему пути не относится.

## N/A-VULKAN / N/A-METAL / N/A-LINUX-ONLY

- Vulkan/Metal: в CUDA-срезе не встречались как отдельные generic-оптимизации;
  metal turbo файлы - часть 1a285c4d1 (уже в N/A-TURBOQUANT/FORK-INTERNAL).
- LINUX-ONLY: madvise/expert-residency POSIX-подходов в форке не найдено
  (`grep_offload.txt`); Windows-аналог (VirtualUnlock и пр.) форк не писал.

## MoE CPU-offload: итог спецпроверки c

Отдельных CUDA-оптимизаций mul_mat_id / expert offload / overlap / async / pinned
memory (cudaMallocHost) для MoE-on-CPU в форке НЕТ. Есть только размещение DFlash
draft-весов. Единственный близкий механизм - batch async копирования (BE-297a1cd83,
PORTABLE, см. 02-applicable.md приоритет 3).
