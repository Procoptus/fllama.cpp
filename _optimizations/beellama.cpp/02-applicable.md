# 02-applicable.md
# Применимые оптимизации: что править, где, как

Честная оговорка: beellama - форк про ROCm/KVarN/TurboQuant/DFlash. Полноценных
generic-CUDA speedup'ов для нашего сетапа (CUDA + qwen4exp + f16 KV + Q3_K_XL)
в нём найдено мало: один прямой speedup (GDN exp2), одна CONFIG-проверка
(OpenMP, знание из CI-фикса форка), один PORTABLE-механизм (async-batch
копирований) и один correctness-guard. Всё остальное - в
[04-not-applicable.md](04-not-applicable.md).

## Приоритет 1. BE-e275191e8 - exp2 SFU fast path для GDN gate decay (CUDA, generic)

### Почему это бьёт именно в наш случай

- Наша модель qwen4exp использует gated delta-net: op `GGML_OP_GATED_DELTA_NET`
  строится в [src/models/delta-net-base.cpp](src/models/delta-net-base.cpp:399),
  на CUDA исполняется ядром `gated_delta_net_cuda`
  ([ggml/src/ggml-cuda/gated_delta_net.cu](ggml/src/ggml-cuda/gated_delta_net.cu:84)).
- Decay-вентиль `g` на each шаг умножается на `expf(*g_t)` / `expf(g_t[i])`:
  это hot loop и в TG (каждый токен, все GDN-слои), и в inner loop prefill.
- `expf` на CUDA компилируется в sequence multiply + ex2 + range reduction;
  `exp2f` - одна инструкция SFU `ex2.approx` (цитата механики из сообщения
  коммита форка). Замена `expf(x)` -> `exp2f(x * 1.442695041f)` (1/ln2)
  убирает лишнюю арифметику из горячего прохода по состояниям S_v x S_v.
- Проверено grep по нашему дереву: `expf(*g_t)` строка 85, `expf(g_t[i])`
  строки 118 и 132 [gated_delta_net.cu](ggml/src/ggml-cuda/gated_delta_net.cu:84) -
  наше дерево НЕ содержит этой оптимизации.
- Привязки к KVarN/TurboQuant в диффе нет - файл правится только этот,
  строки только generic-ядер (спецпроверка a: VALUABLE).

### Что портить

Из commit e275191e8 (форк):

```cpp
// exp2 is a single SFU instruction (ex2.approx) vs expf's multiply + ex2 + range reduction
#define GDN_EXPF(x) exp2f((x) * 1.442695041f)
```

- Добавить макрос в начало [ggml/src/ggml-cuda/gated_delta_net.cu](ggml/src/ggml-cuda/gated_delta_net.cu:1).
- Заменить `expf(` на `GDN_EXPF(` в трёх местах нашего файла: строки 85, 118, 132.
- Форк менял 6 мест, но 3 из них в его `gated_delta_net_tree_cuda` - tree-ядра
  в нашем дереве нет (grep `tree` в нашем gated_delta_net.cu пуст). Port только
  flat-ядра.

### Что делать

1. Внести правку (4-5 строк).
2. Проверить численную корректность прогоном `test-backend-ops -o GATED_DELTA_NET`
   (ex2.approx имеет относительную погрешность ~1e-6, для decay-вентиля f32 это
   безопасно; у форка assert'ов точности в диффе нет - опираться на свой тест).
3. Замерить TG/PP до/после на нашей модели. Эффект не заявлен автором в цифрах:
   "(оценка: доли процента на слой, суммарно по GDN-слоям заметно только в TG
   профиле; не измерено)".

## Приоритет 2. BE-f7ca480ed (знания) - CONFIG: проверить, что OpenMP реально в сборке

### Почему бьёт

- У нас `ot="(.ffn_.*exp|per_layer_token_embd)=CPU"`: каждый токен TG гоняет
  CPU-graph-splitters (mul_mat_id экспертов + PLE) рядом с CUDA. Это partial
  offload - ровно конфигурация, где потеря OpenMP стоит дороже всего.
- Механика из сообщения коммита форка (замер автора, issue #111 форка, ROCm/CPU
  контекст): без `GGML_USE_OPENMP` `ggml_graph_compute` создаёт одноразовый
  threadpool (spawn/join n_threads-1 OS-нитей) на КАЖДЫЙ CPU graph split -
  фиксовано ~0.5 мс; "negligible during prefill but dominates token generation
  for partially offloaded models (-35..40% tg with --n-cpu-moe)".
- Наш риск идентичен по природе: MSVC-сборка, `find_package(OpenMP)` молча
  WARNING-ится при ненайденном компиляторном окружении; GGML_OPENMP=ON в
  cmdline не гарантирует, что флаг дожил до компиляции.

### Что портить

Ничего из кода. Только приём проверки из CI-фикса: жёсткий assert
`GGML_OPENMP_ENABLED` после configure.

### Что делать

1. В нашем build cache проверить: `OpenMP_CXX_FOUND=TRUE`, `GGML_USE_OPENMP`
   в ggml-runtime (после сборки: `dumpbin /exports ggml.dll | findstr omp`,
   либо verbose-лог llama-server: строка "CPU: ... OpenMP" / бэкенд-фичи).
2. Если OpenMP потерян - пересобрать из Developer PowerShell (полный vcvars),
   не из голой cmd.
3. Это CONFIG-only: правок в исходники нет.

## Приоритет 3. BE-297a1cd83 - PORTABLE: batch async D2D копирования GDN-состояний

### Почему бьёт

- Механика: 96 отдельных `ggml_backend_tensor_copy` (в каждом своя
  `cudaStreamSynchronize` - полная синхронизация стрима) заменены на очередь
  `ggml_backend_tensor_copy_async` + один `ggml_backend_synchronize` в конце.
  Каждый лишний sync на TG-пути - это сериализация CPU/GPU.
- У нас `copy_cell` нет (grep по src/*.cpp: 0). Наша запись состояния
  рекуррентной памяти идёт через `llama_memory_state_*` /
  [llama-memory-recurrent.cpp](src/llama-memory-recurrent.cpp) - при
  restore/checkpoint (ctx-checkpoints 3) тоже несколько отдельных
  `ggml_backend_tensor_copy` по слоям.

### Что портить

Идею, не diff: сгруппировать покадровые копирования состояний в async-очередь
на выделенном copy-backend c одним sync. Файл-прототип форка:
`src/llama-memory-recurrent.cpp` (+40/-4), заголовок `llama-memory-recurrent.h`.

### Что делать

1. Профилировать, сколько копирований происходит на наш restore/checkpoint
   (наш путь: state_seq_get_data/state_seq_cpy в llama-memory-recurrent.cpp).
2. Если копирований много и они последовательно sync'ятся - переписать по
   образцу. Требуется ручной разбор расхождений API, поэтому PORTABLE, а не
   CUDA-APPLICABLE.
3. Эффект "(оценка, не измерено)" - форк цифр в сообщении не привёл.

## Приоритет 4. BE-86ec665fa - correctness guard MMVF/MMF (честный вердикт)

### Что это честно

НЕ speedup. Спецпроверка h: diff добавляет валидацию `src1->nb` (шаги RHS
должны быть кратны `2*sizeof(float)`, иначе float2-чтения в mmvf невалидны) +
проверку `vals_per_T` в mmf.cu + `ggml_cuda_mmvf_rhs_compatible()` в mmvf.cu +
расширенный тест k_v_rhs в test-backend-ops. На невыровненном RHS ядро просто
не выбирается (fallback). Один-в-один "может лишь чуть замедлить/консервативнее
роутить", ускорения там нет и автором не заявлялось.

### Почему всё равно стоит взять

- Наш CPU-offload путь даёт нестандартные раскладки src1 (срезы PLE-таблиц,
  gathered-вектора). Guard защищает от реальных UB-чтений в mmvf на CUDA.
- Порт лёгкий: 2 файла ggml-cuda + тест.

### Что делать

Перенести guard как есть (ggml-cuda.cu call-sites с `src1->nb`, mmf.cu loop по
GGML_MAX_DIMS, mmvf.cu helper). Прогнать `test-backend-ops -o MUL_MAT_ID` и
`-o MUL_MAT` с rhs-вариантами.

## Не применять

- Всё из [04-not-applicable.md](04-not-applicable.md) (KVarN, TurboQuant,
  DFlash/spec, ROCm, preserve-мегамассы).
- `ac792986d` (4D-state compat): наше дерево уже на 3D snapshot API, обратная
  совместимость со старым 4D layout нам не нужна.
- `91357ddcc` (topk K<=64): наш сэмплер top-k=20; фикс решал проблему DFlash
  verifier с Gemma4 top_k=64.
