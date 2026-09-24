# 06-build-flags-audit.md
# Аудит флагов сборки: что форк меняет / что проверять у нас

## 1. Флаги нашего сетапа (база сравнения)

CMake + Ninja, MSVC/cl, Release:

- `GGML_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=86`
- `BUILD_SHARED_LIBS=OFF`
- `GGML_NATIVE=ON`, `GGML_LTO=ON`, `CCACHE=ON`
- `GGML_OPENMP=ON`
- `GGML_SCHED_MAX_COPIES=1`
- `GGML_AVX2=ON, GGML_AVX512=ON, GGML_AVX512_VBMI=ON, GGML_AVX512_VNNI=ON,
  GGML_AVX_VNNI=ON, GGML_BMI2=ON`
- `LLAMA_CUDA_GRAPHS=ON` (compile-time; runtime graphs выключены)
- `LLAMA_FLASH_ATTN=ON`, `GGML_CUDA_FA_ALL_QUANTS=ON`
- Не собираются: Vulkan, ROCm/HIP, Metal, RPC, mtmd - ок, форк их не трогает
  в интересующих нас путях.

Ни один флаг из этой матрицы форком не переопределяется и не конфликтует
с правками из [02-applicable.md](02-applicable.md).

## 2. Что форк меняет в сборке (и почему нам не нужно)

| Флаг/механизм форка | ID | Вердикт для нас |
|---|---|---|
| `GGML_HALF_QUANTS` build mode (урезание FA pair matrix) | BE-a596a5c72 + BE-5eaba174c + BE-df8933c26 | N/A. Матрица FA-пар нужна нам полной (FA_ALL_QUANTS=ON + f16 KV). Урезание форка сделано ради времени сборки их turbo/kvarn типов |
| CUDA graphs props check cache | BE-1eaea4494 = upstream c5ce4bc22 | UPSTREAM; moot - runtime graphs OFF |
| KVarN/TurboQuant компиляционные опции | семья BE-1010625c5/BE-098e00d34 | N/A-TURBOQUANT/KVARN |
| ROCm/HIP релизные джобы, vswhere в CI | BE-f7ca480ed/BE-db8119cfb/BE-94605e9fe | Наш CI не он; но см. п.3 |

## 3. CONFIG-вывод из форка: OpenMP на MSVC может молча теряться

Главная находка аудита сборки (детали: 02-applicable.md приоритет 2):

- В форке CI-джобы windows-cpu вызывали `vcvarsall.bat` по хардкод-пути VS2022
  Enterprise, которого нет на windows-2025 image. Вызов падал молча, MSVC-окружение
  не поднималось, `find_package(OpenMP)` не резолвил OpenMP_C/OpenMP_CXX, а это
  только CMake WARNING - релиз уезжал без `GGML_USE_OPENMP`.
- Цена (цитата сообщения BE-f7ca480ed): без OpenMP `ggml_graph_compute` создаёт
  одноразовый threadpool на каждый вызов без прикреплённого пула; llama-server
  пул не прикрепляет; фиксовано ~0.5 мс на CPU graph split; "dominates token
  generation for partially offloaded models (-35..40% tg with --n-cpu-moe)".
- Наш прогон partial-offload (MoE/experts + PLE на CPU): профиль риска тот же.

### Чек-лист для нашего дерева

1. После configure: в `CMakeCache.txt` наличие `OpenMP_CXX_FOUND:BOOL=TRUE`
   и отсутствие warning "OpenMP not found".
2. Собирать из Developer PowerShell/Command Prompt (полный vcvars-контекст),
   не из голой cmd.
3. После сборки подтвердить факт: presence omp-символов в ggml-библиотеке
   (`dumpbin /exports` для shared; для static - лог бэкенда GGML при старте:
   CPU-бэкенд печатает список фич, OpenMP должен быть среди них).
4. Правка кода не требуется - CONFIG-only.

## 4. Итог по флагам

- Ни один флаг нашей матрицы менять не нужно (включая GGML_CUDA_FA_ALL_QUANTS=ON:
  форк урезает матрицу только в своём HALF_QUANTS-режиме, это не "оптимизация"
  для нас).
- Единственный CONFIG-пункт: верификация, что OPENMP реально попал в бинарь
  (см. п.3).
- Отдельных speedup-флагов (novel cflags, arch overrides, PGO/LTO-трюков)
  форк не вводит.
