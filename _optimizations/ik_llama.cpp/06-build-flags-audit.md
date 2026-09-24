# Аудит флагов сборки

Источник истины - `build_avx_512/CMakeCache.txt`, а не командная строка.
Дата аудита: 2026-09-02, HEAD `ead916c05`, MSVC 14.44.35207 + Ninja. См. также раздел «Расхождения с аудитом».

## Найденные ошибки

| Флаг в команде | Реальность | Доказательство | Действие |
|---|---|---|---|
| `-DGGML_CUDA_GRAPH_OPT=1` | **не CMake-опция**, в кэше `UNINITIALIZED`. Runtime переменная окружения | `ggml/src/ggml-cuda/ggml-cuda.cu:4441` - `getenv("GGML_CUDA_GRAPH_OPT")`; `build_avx_512/CMakeCache.txt:686` UNINITIALIZED | убрать из cmake; задавать `set GGML_CUDA_GRAPH_OPT=1` в батнике (пользователь уже делает) |
| `-DGGML_CUDA_USE_GRAPHS=ON` | **не CMake-опция**, `UNINITIALIZED`. Правильное имя `GGML_CUDA_GRAPHS` | `build_avx_512/CMakeCache.txt:701` UNINITIALIZED; определение `GGML_CUDA_USE_GRAPHS` генерируется из `GGML_CUDA_GRAPHS` - `ggml/src/ggml-cuda/CMakeLists.txt:134`, `:135` | удалить как мёртвый: цель всё равно достигнута, `GGML_CUDA_GRAPHS=ON` в кэше `:683` и форсится `CMakeLists.txt:169` |
| `-DGGML_CUDA_F16=ON` | **флага не существует** в этом дереве | `build_avx_512/CMakeCache.txt:668` UNINITIALIZED; в `ggml/CMakeLists.txt:199-213` такого варианта нет | удалить |
| `-DGGML_FMA=ON` | объявлена только под `if (NOT MSVC)` -> игнорируется | `ggml/CMakeLists.txt:163`; кэш `:713` UNINITIALIZED | удалить; FMA включается через `/arch:AVX512` - `ggml/src/ggml-cpu/CMakeLists.txt:249` |
| `-DGGML_F16C=ON` | объявлена только под `if (NOT MSVC)` -> игнорируется | `ggml/CMakeLists.txt:165`; кэш `:710` UNINITIALIZED | удалить; см. выше |
| `-DGGML_AVX2=ON` | передан дважды | команда пользователя | дедуплицировать |
| `-DGGML_AVX512_BF16=ON` | **ПРИМЕНЯТЬ НЕЛЬЗЯ**: под cl.exe не компилируется, CPU без AVX512-BF16 | C2440 в `sgemm.cpp:374/377`; блок `sgemm.cpp:372-385` без `!defined(_MSC_VER)`; Alder Lake (Model 151) не имеет бита AVX512_BF16 | убрать; в `build_avx_512` уже стоит `OFF` (`CMakeCache.txt:599`). См. раздел BF16 ниже |

## Что реально принято сборкой (по кэшу)

| Опция | Значение | Строка кэша |
|---|---|---|
| `CMAKE_CXX_COMPILER` | MSVC 14.44.35207 | :590 |
| `GGML_AVX2` | ON | :593 |
| `GGML_AVX512` | ON | :596 |
| `GGML_AVX512_VBMI` | ON | :602 |
| `GGML_AVX512_VNNI` | ON | :605 |
| `GGML_AVX_VNNI` | ON | :608 |
| `GGML_BMI2` | ON | :626 |
| `GGML_CCACHE` | found | :635 |
| `GGML_CPU_ALL_VARIANTS` | OFF | :644 |
| `GGML_CPU_REPACK` | ON | :659 |
| `GGML_CUDA_FA` | ON | :671 |
| `GGML_CUDA_FA_ALL_QUANTS` | ON | :674 |
| `GGML_CUDA_GRAPHS` | ON | :683 |
| `GGML_LTO` | ON | :755 |
| `GGML_NATIVE` | ON | :785 |
| `GGML_OPENMP` | ON | :803 |
| `GGML_SCHED_MAX_COPIES` | 1 | :845 |

Замечание про автоопределение: `ggml/src/ggml-cpu/cmake/FindSIMD.cmake:92`
прощупывает только AVX / AVX2+FMA / AVX_VNNI / AVX512. `AVX512_VBMI`,
`AVX512_VNNI`, `AVX512_BF16` автоматически не определяются - их нужно указывать
вручную, что и делается.

## Очищенная команда конфигурирования

```bat
cmake -B build_avx_512 ^
  -G Ninja ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DCMAKE_CUDA_ARCHITECTURES=86 ^
  -DCMAKE_CUDA_STANDARD=17 ^
  -DBUILD_SHARED_LIBS=OFF ^
  -DLLAMA_CURL=OFF ^
  -DGGML_NATIVE=ON ^
  -DGGML_LTO=ON ^
  -DGGML_CCACHE=ON ^
  -DGGML_OPENMP=ON ^
  -DGGML_SCHED_MAX_COPIES=1 ^
  -DGGML_AVX2=ON ^
  -DGGML_AVX512=ON ^
  -DGGML_AVX512_VBMI=ON ^
  -DGGML_AVX512_VNNI=ON ^
  -DGGML_AVX512_BF16=OFF ^
  -DGGML_AVX_VNNI=ON ^
  -DGGML_BMI2=ON ^
  -DGGML_CUDA=ON ^
  -DGGML_CUDA_GRAPHS=ON ^
  -DGGML_CUDA_FA=ON ^
  -DGGML_CUDA_FA_ALL_QUANTS=ON
```

Сборка:

```bat
cmake --build build_avx_512 --config Release -j 16 --clean-first --target llama-server
```

Отличия от исходной команды:

- удалено: `GGML_CUDA_GRAPH_OPT=1`, `GGML_CUDA_USE_GRAPHS=ON`, `GGML_CUDA_F16=ON`,
  `GGML_FMA=ON`, `GGML_F16C=ON`, дубликаты `GGML_AVX2=ON` и `GGML_F16C=ON`
- добавлено: ничего
- отменено (2026-09-02): `GGML_AVX512_BF16=ON` - не применять (см. раздел BF16 ниже)
- поведение всех удалённых флагов не менялось: они либо не существовали, либо
  игнорировались под MSVC, либо дублировали уже принятые значения

## Переменные окружения при запуске

| Переменная | Значение | Эффект | Где читается |
|---|---|---|---|
| `GGML_CUDA_GRAPH_OPT` | `1` | оптимизация CUDA graphs | `ggml/src/ggml-cuda/ggml-cuda.cu:4441` (уже задано пользователем) |
| `GGML_CUDA_DISABLE_GRAPHS` | не задавать | полное отключение graphs | `ggml/src/ggml-cuda/common.cuh:1258` |
| `LLAMA_GRAPH_REUSE_DISABLE` | не задавать | отключение переиспользования графа, даёт лог | `src/llama-context.cpp:279-284` |
| `GGML_CUDA_ENABLE_UNIFIED_MEMORY` | не задавать | unified memory | `ggml/src/ggml-cuda/ggml-cuda.cu:142`, `:4890` |

## Конфигурационные конфликты (не сборка, а запуск)

| Конфликт | Детали |
|---|---|
| `--fit` + ручной `-ot` | `common/fit.cpp:483` бросает `common_params_fit_exception`, если `tensor_buft_overrides` уже задан. Обход: `-ncmoe` вместо `-ot` regex (см. `02-applicable.md`) |
| `--load-mode mlock` + `--lazy-mode on` | **конфликта нет**, обработано upstream: `src/llama-model-loader.cpp:1403` («read_lazy also requires mmap») и `:1636` («locking a lazy tensor would fault all of it in, which is what lazy avoids») -> `if (lmlocks && !lazy.has(cur))` |
| `-ot` на CPU + CUDA graphs | **конфликта нет**: CPU-оффлоадные `MUL_MAT_ID` узлы не попадают в CUDA graph; cross-device CPY идёт через events, не syncs (`ggml/src/ggml-cuda/ggml-cuda.cu:2555-2570`) |

## BF16: отмена рекомендации (2026-09-02)

Рекомендация `-DGGML_AVX512_BF16=ON` ошибочна и отменена. Сборка с ней падает, и даже
гипотетически успешная сборка была бы бесполезна на этом CPU.

### 1. Под MSVC cl.exe не компилируется никогда

Блок [`sgemm.cpp:372-385`](../../ggml/src/ggml-cpu/llamafile/sgemm.cpp) закрыт только
`#if defined(__AVX512BF16__)`, guard'а `!defined(_MSC_VER)` нет:

```cpp
#if defined(__AVX512BF16__)
template <> inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);      // 374: C2440
}
template <> inline __m256bh load(const ggml_bf16_t *p) {
    return (__m256bh)_mm256_loadu_ps((const float *)p);      // 377: C2440
}
#endif
```

`_mm512_loadu_ps` возвращает `__m512`. В GCC/Clang каст `__m512 -> __m512bh` допустим, в MSVC
`__m512bh` - отдельный struct (zmmintrin.h / immintrin.h) с конструкторами только из
`__m512i`/`__m256i` -> `error C2440`. Следующие два перегруженных `load` (через
`_mm512_cvtne2ps_pbh` / `_mm512_cvtneps_pbh`) не падают, они возвращают нативный `__m512bh`.

Работающий аналог в дереве существует: макрос `m512bh()` в
[`ggml-cpu-impl.h:33`](../../ggml/src/ggml-cpu/ggml-cpu-impl.h) (`#if defined(_MSC_VER)
#define m512bh(p) p`), им защищён `ggml_vec_dot_bf16` в [`vec.cpp:139`](../../ggml/src/ggml-cpu/vec.cpp).
Вендорный sgemm.cpp его не использует.

Блок вендорный (llamafile, в файле с 2024), в upstream с коммита `2cd43f490` (2024-12-24, PR #10714),
MSVC-guard за всё это время не добавлялся. В CI upstream дорожка `cl.exe + __AVX512BF16__`
не собирается ни одним job'ом (`.github/workflows/build-cpu.yml`; avx512-тест с Intel SDE
закомментирован, L205-209). То есть это не наша поломка и не «когда-то сломано и починено»,
а не покрытая upstream дорожка.

### 2. На i7-12700K флаг бесполезен и опасен даже при успешной сборке

- CPU: Alder Lake-S, Family 6 Model 151 Stepping 2. Аппаратно AVX512-BF16 нет (появился только
  в Sapphire Rapids / Zen4). Есть F/CD/BW/DQ/VL/VBMI/VBMI2/VNNI, AVX-VNNI, BMI2, FMA.
- Выбор bf16-ядра чисто compile-time: [`sgemm.cpp:3893`](../../ggml/src/ggml-cpu/llamafile/sgemm.cpp)
  `case GGML_TYPE_BF16:` -> `#if defined(__AVX512BF16__)`. Runtime-условия нет.
- Runtime-защиты в нашей конфигурации нет: `ggml_backend_cpu_x86_score()`
  ([`cpu-feats.cpp:263`](../../ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp)), который возвращает 0 при
  отсутствии CPUID-бита, участвует в сборке только при `GGML_BACKEND_DL=ON`; у нас `OFF`
  ([`ggml-cpu/CMakeLists.txt:373-379`](../../ggml/src/ggml-cpu/CMakeLists.txt)).
- `ggml_cpu_has_avx512_bf16()` ([`ggml-cpu.c:3683`](../../ggml/src/ggml-cpu/ggml-cpu.c)) - чистый
  compile-time (`#if defined(__AVX512BF16__) return 1`), единственный читатель - печать списка
  фич ([`ggml-cpu.cpp:572`](../../ggml/src/ggml-cpu/ggml-cpu.cpp)).
- CPUID-бит в коде: `AVX512_BF16() { return f_7_1_eax[5]; }` ([`cpu-feats.cpp:84`](../../ggml/src/ggml-cpu/arch/x86/cpu-feats.cpp)).

Итог: при обходе ошибки компиляции `VDPBF16PS` выполнялся бы безусловно при первом же BF16
matmul -> illegal instruction.

### 3. Флаг не включается сам по себе

`GGML_NATIVE=ON` через `FindSIMD.cmake` (L93-118) пробует только AVX / AVX2+FMA / AVX_VNNI /
AVX512. Проб на VBMI / AVX512-VNNI / BF16 нет, поэтому BF16 мог попасть в сборку только из явного
`-DGGML_AVX512_BF16=ON`. `GGML_NATIVE=ON` не опасен в этом смысле.

### 4. Верифицирующий тест (выполнен 2026-09-02)

- `cmake -B build_avx_512 -DGGML_AVX512_BF16=OFF` -> Configuring done / Generating done, ошибок нет.
- `build_avx_512/CMakeCache.txt:599` -> `GGML_AVX512_BF16:BOOL=OFF`.
- `-D__AVX512BF16__` в `build_avx_512/build.ninja` -> 0 совпадений.
- `ninja -C build_avx_512 ggml/src/CMakeFiles/ggml-cpu.dir/ggml-cpu/llamafile/sgemm.cpp.obj` ->
  exit code 0, C2440 нет.

### 5. Практическое

`build_avx_512/CMakeCache.txt` переживает перезагрузку с включённым BF16, поэтому при следующей
конфигурации нужно явное `-DGGML_AVX512_BF16=OFF` (в текущем кэше уже стоит OFF) либо новый
каталог сборки. Полная сборка после этого не проверялась, проверена только цель sgemm.cpp.

## Расхождения с аудитом (2026-09-02)

Три пункта данной команды противоречат аудиту дерева HEAD `ead916c05`.

1. `-DCMAKE_CUDA_ARCHITECTURES=86` перетирается. `GGML_NATIVE=ON` форсит `CMAKE_CUDA_ARCHITECTURES
   "native"` ([`ggml-cuda/CMakeLists.txt:27-28`](../../ggml/src/ggml-cuda/CMakeLists.txt)). Нужно
   выбрать одно: либо `-DGGML_NATIVE=OFF` + явный `86`, либо оставить NATIVE и убрать `86`.
   Для RTX 3080 native даёт `86-real`/`86-virtual`, что корректно.
2. `-DLLAMA_CURL=OFF` - устаревший алиас, CURL-подсистема удалена:
   `llama_option_depr(WARNING LLAMA_CURL)` ([`CMakeLists.txt:195`](../../CMakeLists.txt)),
   machinery в [`CMakeLists.txt:174-184`](../../CMakeLists.txt). Флаг игнорируется, убрать.
3. `-DGGML_CUDA_F16=ON` уже помечен выше как несуществующий - подтверждено: в дереве 0 совпадений.

### Что можно добавить (существующие опции, ускоряют сборку llama-server)

| Option | Определение | Default |
|---|---|---|
| `-DLLAMA_BUILD_TESTS=OFF` | `CMakeLists.txt:132` | `${LLAMA_STANDALONE}` |
| `-DLLAMA_BUILD_EXAMPLES=OFF` | `CMakeLists.txt:134` | `${LLAMA_STANDALONE}` |
| `-DLLAMA_BUILD_APP=OFF` | `CMakeLists.txt:136` | `${LLAMA_STANDALONE}` |
| `-DGGML_BUILD_TESTS=OFF` | `ggml/CMakeLists.txt:279` | `${GGML_STANDALONE}` |
| `-DGGML_BUILD_EXAMPLES=OFF` | `ggml/CMakeLists.txt:280` | `${GGML_STANDALONE}` |

`LLAMA_BUILD_UI` (`CMakeLists.txt:137`) уже OFF. `LLAMA_BUILD_TOOLS` (`CMakeLists.txt:133`) НЕ
отключать: в нём резидент llama-server (проверка связи не выполнялась, но риск высокий).
`-DGGML_CUDA_FA_ALL_QUANTS=ON` заметно удлиняет компиляцию CUDA (glob `fattn-vec*.cu`,
`ggml-cuda/CMakeLists.txt:115`): оставлять только если нужны все варианты KV-квантов в FA.
