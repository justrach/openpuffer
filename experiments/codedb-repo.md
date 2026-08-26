# codedb real-repository retrieval gate

This secondary gate uses vectors from the actual `justrach/codedb` repository
instead of the clustered generator. It answers two different questions:

1. **ANN recall:** does OpenPuffer agree with exact cosine top-k over the same
   vectors?
2. **Semantic retrieval:** does exact dense retrieval find a file changed by
   the commit used as the query?

Do not use semantic hit rate as the ANN gate. A model can miss the relevant
file even when the ANN implementation reproduces exact dense ranking perfectly.

## Dataset contract

- model: `Qwen/Qwen3-Embedding-0.6B`
- dimensions: 512, normalized `float32`
- corpus: 737 secret-filtered tracked-file cards from `justrach/codedb`
- corpus cache SHA-256:
  `a1bcf947eff13285cbaf678cb6fa89e0d75628a1d239072a66b6d89be0acf603`
- queries: 100 recent non-merge codedb commit subjects that touch at least one
  path present in the corpus cache
- query prefix:
  `Instruct: Given a software engineering issue, retrieve the code files that must be edited to resolve the issue\nQuery:<subject>`
- k: 24
- ANN recall floor: 0.99 against exact cosine top-24

The `.npz` corpus and query caches are external artifacts and are never
committed. If the corpus cache is absent (or `--refresh-corpus` is used), the
tool builds tracked-file cards itself while excluding `.env`, PEM/key files,
credentials, symlinks, binary files, and files over 128 KiB before sending
text to the GPU embedding service. It never loads `.env` files and has no CPU
embedding fallback.

Example:

```sh
python3 tools/codedb_repo_bench.py \
  --repo /path/to/codedb \
  --corpus-cache /path/to/codedb-files-qwen3-0.6b-d512.npz \
  --query-cache /path/to/codedb-queries-qwen3-0.6b-d512.npz \
  --gateway http://127.0.0.1:9000 \
  --ef-sweep 48,64,96,128 \
  --rerank-mults 1,2,4 \
  --repeats 3 \
  --require-quality
```

## 2026-08-26 baseline and knob hill climb

The interleaved localhost HTTP sweep prevents configuration order and machine
drift from biasing the default/control. Exact dense semantic quality was:

| corpus | queries | exact hit@24 | exact MRR@24 | exact nDCG@24 |
|---:|---:|---:|---:|---:|
| 737 | 100 | 0.8000 | 0.5026 | 0.3980 |

Selected real-repository configurations:

| ef | rerank | HTTP p50 | ANN recall@24 | top-1 agreement | gold hit@24 | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 48 | 1 | 0.402 ms | 0.9900 | 0.990 | 0.8000 | reject: fails 20k synthetic recall |
| **48** | **2** | **0.403 ms** | **0.9958** | **0.990** | **0.8000** | **codedb caller profile** |
| 64 | 2 | 0.407 ms | 0.9971 | 0.990 | 0.8000 | quality margin, slower |
| 128 | 4 | 0.419 ms | 0.9992 | 1.000 | 0.8000 | global default/control |

The fixed 20k×512 clustered benchmark was then run as fresh paired processes
so RSS and the arena do not accumulate across an ef sweep:

| profile | run 1 p50 | run 2 p50 | recall@24 | build | RSS |
|---|---:|---:|---:|---:|---:|
| default `ef=128, rerank=4` | 0.090 ms | 0.091 ms | 1.0000 | 5453–5573 ms | 185 MiB |
| tuned `ef=48, rerank=2` | **0.058 ms** | **0.059 ms** | **0.9992** | 5488–5536 ms | 184 MiB |

This is a 34–36% direct ANN p50 reduction with the recall floor, build-time,
and RSS gates intact. `ef=48, rerank=1` is not safe: its synthetic recall was
only 0.9469, even though the 732-vector real-repository sample reached 0.9900.

The result is an explicit codedb request profile, not a change to the generic
OpenPuffer server default. Callers should send `ef=48` and `rerank_mult=2` for
this 512D/k=24 workload and retain the existing global default for unmeasured
dimensions and distributions.

## Kernel experiments

| id | hypothesis | source change | paired synthetic p50 | synthetic recall@24 | real-repo recall@24 | verdict | notes |
|---|---|---|---:|---:|---:|---|---|
| R001 | use a 384D prefix only for 512D upper-layer routing, then restore the full 512D distance at layer 0 | reverted | control 0.058/0.059 ms; candidate 0.057/0.058 ms | 0.9992 | 0.9950 (732-cache) | discard | Recall held, but the ~1–2% latency change did not clear the 4% noise/keep bar. No source change retained. |
