# 03-already-upstream.md
# Эквивалент уже есть в upstream (и в нашей базе 8887a48f0)

Метод проверки: форк нёс коммиты upstream PR'ов; наличие в нашей базе
проверено `git merge-base --is-ancestor <upstream_sha> 8887a48f0` в клоне
(результат: `D:\dev\llama_compiling\bee_audit\anc.txt`, все 4 - IN).

| ID (форк) | PR | Upstream-эквивалент | Предмет | Примечание |
|---|---|---|---|---|
| [BE-1eaea4494](https://github.com/Anbeeld/beellama.cpp/commit/1eaea4494) | #21472 | `c5ce4bc22` (ancestor) | make cuda graphs props check faster: переписан props-check через кеш, -128 строк в common.cuh + ggml-cuda.cu | Moot для нас: CUDA graphs выключены на runtime (специально проверено). Даже не портируя, ничего не теряем |
| [BE-336e225f0](https://github.com/Anbeeld/beellama.cpp/commit/336e225f0) | #21676 | `009a11332` (ancestor) | ggml: check return value of CUB calls in argsort and top-k (все они возвращают cudaError_t) | Только корректность; наш argsort/top-k (QSA indexer) уже с чеками |
| [BE-66c4f9ded](https://github.com/Anbeeld/beellama.cpp/commit/66c4f9ded) | #21168 | `66c4f9ded` (ancestor) | ggml-cuda: ds_read_b128 для q4_0/q4_1 mmq-ядер | Косвенно закрывает спецпроверку f: это единственное найденное generic k-quant mmq-ускорение, и оно уже в базе. Q3_K-специфичных ускорений в форке нет |
| [BE-e34f04215](https://github.com/Anbeeld/beellama.cpp/commit/e34f04215) | #21665 | `e34f04215` (ancestor) | CUDA: fuse muls | Fused mul-путь уже в базе |

Вывод: переносить из этих четырёх нечего.
