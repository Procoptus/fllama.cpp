# 03-already-upstream.md - эквивалент уже есть в нашем дереве

## CL-4db9548a8 - disable shift/cache-reuse for IMROPE models (n_pos_per_embd > 1)

Ссылка: https://github.com/fewtarius/CachyLLama/commit/4db9548a8c421da633e1124b4d4110b82306803e (2026-07-03)

### Что делает коммит

Добавляет guard `hparams.n_pos_per_embd() > 1` в `get_can_shift()` рядом с
существующим STEP35-guard: для IMROPE/MROPE моделей seq_add()/seq_div()
assert'ят `n_pos_per_embd() == 1` (не умеют сдвигать многомерные
kv_cell_ext), но llama_memory_can_shift() этого не учитывала, и путь
`--cache-reuse` доводил до GGML_ABORT.

### Доказательство, что у нас уже есть

В нашем дереве [src/llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189)
`get_can_shift()` содержит оба guard'а - и STEP35, и
`hparams.n_pos_per_embd() > 1` (проверено построчно). Эквивалент уже в
upstream, переносить нечего.

### Почему это важно именно нам (защита работает)

Наша модель qwen4exp - IMROPE:
- [src/llama-model.cpp:2960](src/llama-model.cpp:2960): LLM_ARCH_QWEN4EXP -> LLAMA_ROPE_TYPE_IMROPE
- [src/llama-hparams.cpp:260](src/llama-hparams.cpp:260): MROPE/IMROPE -> n_pos_per_embd = 4
- [src/models/qwen4exp.cpp:630](src/models/qwen4exp.cpp:630), [:823](src/models/qwen4exp.cpp:823): ggml_rope_multi
- конвертер [conversion/qwen4exp.py:12](conversion/qwen4exp.py:12): миксин `_Qwen35MRopeMixin`

Т.е. без этого guard'а любой запуск с `--cache-reuse` на нашей модели падал бы
на assert в seq_add. У нас guard активен: ctx_shift и cache_reuse для qwen4exp
запрещены штатно. Prompt cache (LCP prefix reuse) не затрагивается - отключается
только опциональный chunk-shifting.

## Итог

| ID | Статус | Доказательство |
|---|---|---|
| [CL-4db9548a8](https://github.com/fewtarius/CachyLLama/commit/4db9548a8c421da633e1124b4d4110b82306803e) | UPSTREAM | [src/llama-kv-cache.cpp:1189-1198](src/llama-kv-cache.cpp:1189) |
