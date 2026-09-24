# 06-build-flags-audit.md - аудит флагов сборки: что форк меняет и что нам с этого

Данные: `git diff 3466812d1..8dca17b4d -- '**/CMakeLists.txt' '**/*.cmake' 'cmake/*'`
(read-only clone форка) + наш `build_avx_512/CMakeCache.txt` (MSVC 14.44 + Ninja,
Release, CUDA 13.3, arch 86).

## 1. Что форк вообще менял в cmake (все non-merge коммиты, затронувшие build-файлы)

| Коммит | Файл | Суть | Нам применимо? |
|---|---|---|---|
| [CL-90cbb2e7e](https://github.com/fewtarius/CachyLLama/commit/90cbb2e7e8d27f45f8db7e2b70c5f4c5d678f90b), [CL-b9ed083e4](https://github.com/fewtarius/CachyLLama/commit/b9ed083e41b5ae79d942e6d0e66bd1f2e7b30429), [CL-da9a6da82](https://github.com/fewtarius/CachyLLama/commit/da9a6da82b6a3d43361d915bbac7b4d5d6d022a6) (часть), корневой CMakeLists | trim сборки: LLAMA_BUILD_EXAMPLES=OFF, LLAMA_BUILD_APP=OFF, app/ не собирается | Нет. Это server-only trim под профиль форка; у нас tools/tests собираются штатно, trim не даёт эффекта на скорость рантайма |
| [CL-4d8fae05b](https://github.com/fewtarius/CachyLLama/commit/4d8fae05b4485c8b86646ee971ae68b5b8420912) | LLAMA_BUILD_UI compile flag | Нет (WebUI форка) |
| [CL-6a0db500c](https://github.com/fewtarius/CachyLLama/commit/6a0db500ca1058e06a232c02c50eb5df56b0d151), [CL-56dca0825](https://github.com/fewtarius/CachyLLama/commit/56dca0825e1d0b3a4b5f00a1fc1e59f2c6a900f8), [CL-59da9e100](https://github.com/fewtarius/CachyLLama/commit/59da9e10004cc454d0062d1dfbbd5e0e7b1756f8) | common/: kv-ssd-cache.*, kv-ssd-system-cache.*, host-ram.*, vendor::hash | Нет, изолированно. Только вместе с SSD-подсистемой (02, Приоритет 6) |
| [CL-3ebbc7ea4](https://github.com/fewtarius/CachyLLama/commit/3ebbc7ea43b4e7284e24a1b90d51bbee05d36f9d), [CL-2c8642383](https://github.com/fewtarius/CachyLLama/commit/2c86423831df3eeabd23e51b54bb93d8663c2659) | src/: moe-residency исходники | Нет (Linux madvise, см. 04 группа 5) |
| [CL-bd4f2875b](https://github.com/fewtarius/CachyLLama/commit/bd4f2875b177abd3ada0bdd39318293cdeb3b4d5), [CL-f6c412a61](https://github.com/fewtarius/CachyLLama/commit/f6c412a612b986dcc9107e126f55de07323d9d79) | common/ и ggml-vulkan cmake-правки | Нет (Vulkan / Linux) |
| прочие | tests/CMakeLists.txt (тесты SSD) | Нет (часть BLOCKED-базы) |

Главный вывод: **форк не менял ни одного компиляторного флага, ни одной
опции CUDA/ggml**. Все его cmake-правки - это добавление исходников своих
подсистем и trim профиля сборки. Тюнинг у форка идёт на уровне кода и
Vulkan-шейдеров, не флагов. Портить из cmake нам нечего; применимость
кодовых фиксов описана в [02-applicable.md](02-applicable.md).

## 2. Наш текущий профиль (build_avx_512/CMakeCache.txt) - сверка

| Флаг | Значение | Комментарий |
|---|---|---|
| CMAKE_BUILD_TYPE | Release | OK (/O2) |
| CMAKE_CUDA_ARCHITECTURES | 86 (UNINITIALIZED) | Соответствует RTX 3080 (sm_86). OK |
| GGML_NATIVE | ON | На MSVC это не `-march=native`, а runtime-detect через FindSIMD ([ggml/src/ggml-cpu/cmake/FindSIMD.cmake:92-119](ggml/src/ggml-cpu/cmake/FindSIMD.cmake:92)): компиляция/запуск тестов с /arch:AVX512 и т.д. Проблемы форка CL-5044107be (-march=native на GCC не включает всё) к нам не относятся |
| GGML_AVX2 / BMI2 | ON | 12700K есть |
| GGML_AVX512 / VBMI / VNNI | ON | 12700K с разблокированным AVX-512 (P-cores). MSVC: /arch:AVX512 + ручные __AVX512*__ дефиниции ([ggml/src/ggml-cpu/CMakeLists.txt:254-304](ggml/src/ggml-cpu/CMakeLists.txt:254)) |
| GGML_AVX512_BF16 | OFF | На Alder Lake нет - правильно |
| GGML_AVX_VNNI | ON | Наш собственный патч MSVC-detect (ahead 24); FindSIMD:107 это поддерживает |
| GGML_LTO / OPENMP / CCACHE | ON | OK |
| GGML_SCHED_MAX_COPIES | 1 | Хорошо для parallel=1 (экономия буферов копирования) |
| GGML_CUDA_FA / FA_ALL_QUANTS | ON | Нужно для QSA FA-пути (02, Приоритет 1 - gather требует FA) |
| GGML_CUDA_GRAPHS | ON | +decode TG |
| GGML_CUDA_FORCE_MMQ / FORCE_CUBLAS | OFF | Дефолт, OK для sm_86 |
| GGML_VULKAN / HIP / METAL | OFF | Совпадает с N/A-группами реестра |
| LLAMA_BUILD_MTMD | OFF | Vision не используем |
| LLAMA_OPENSSL | ON, LLAMA_CURL | OFF | OK |

## 3. Итоговые замечания

1. Единственный "флаговый" урок форка (CL-5044107be: молча потерять AVX-512
   при сборке = -30-100% на CPU-оффлоад слоях) у нас закрыт: флаги ISA
   заданы явно, MSVC-detect работает, профиль выше это подтверждает.
2. После портирования кодовых фиксов (02) пересобирать текущим профилем -
   никаких новых cmake-опций включать не требуется.
3. `-ot "(.ffn_.*exp|per_layer_token_embd)=CPU"` зависит от AVX2/AVX-VNNI/
   AVX512 путей в ggml-cpu; при смене машины/микрокода (AVX512-trap)
   пере-проверить FindSIMD-детект: он запускает тестовые бинарники, при
   нестабильном AVX512 молча выключит его и производительность CPU-слоёв
   просядет - это тот же сценарий, что лечил скрипт форка.
