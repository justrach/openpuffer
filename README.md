# openpuffer

An in-memory vector search engine written in Zig 0.17-dev, plus a benchmark
harness that runs it head-to-head against [turbopuffer](https://turbopuffer.com)
using Google `gemini-embedding-2` embeddings as the shared dataset.

## Engine (`src/hnsw.zig`)

- HNSW index over f32 vectors, cosine distance (vectors L2-normalized on insert)
- int8-quantized graph traversal + exact f32 rerank (`searchAdvanced`'s
  `rerank_mult` is the recall/latency knob) — cuts memory traffic 4x at ≥1536 dims
- Explicit `@Vector(16)` dot-product reduction for SIMD without fast-math
- Heuristic neighbor selection with keep-pruned backfill (hnswlib-style)
- Per-layer visited sets (a shared set silently disconnects the graph)

## Commands

```sh
zig build -Doptimize=ReleaseFast

# correctness tests
zig build test

# local ANN vs brute force on random vectors
./zig-out/bin/openpuffer bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128

# full loop: Gemini embeddings -> build local HNSW AND upload to turbopuffer
# -> query both -> latency percentiles + recall@k against exact ground truth
export TURBOPUFFER_API_KEY=...
export GEMINI_API_KEY=...
./zig-out/bin/openpuffer bench-live --namespace openpuffer-bench-5 \
    --n 1000 --queries 40 --dim 1536 --k 10 --ef 256
```

Options for both bench commands: `--n`, `--queries`, `--dim` (MRL output
dimensionality sent to Gemini), `--k`, `--ef`, `--namespace`, `--model`.

## Measured results (M1 Mac, gcp-us-central1 turbopuffer region)

### Head-to-head vs turbopuffer (live gemini-embedding-2 data, k=10)

| dataset | openpuffer p50 | openpuffer recall@10 | turbopuffer p50 | tpuf recall@10 | openpuffer speedup |
|---|---|---|---|---|---|
| 1000 × 1536, ef=256 | **0.72 ms** | 0.960 | 213 ms | 1.000 | ~295x latency |
| 1000 × 3072, ef=256 | **1.01 ms** | 0.965 | 216 ms | 1.000 | ~215x latency |

Caveat: openpuffer is an in-process library (no network hop), turbopuffer a cloud
service measured over WAN — the multiple reflects access model as much as
engine. The apples-to-apples local comparison:

### Local: openpuffer ANN (int8 traversal + f32 rerank) vs exact brute-force scan

| dataset | openpuffer p50 | exact scan p50 | speedup | recall@10 |
|---|---|---|---|---|
| 20k × 1536 random, ef=128 | **1.08 ms** | 3.38 ms | 3.1x | 0.313* |
| 20k × 3072 random, ef=128 | **1.60 ms** | 5.43 ms | 3.4x | 0.272* |
| 1000 × 1536 live embeddings, ef=256 | **0.72 ms** | ~6 ms† | ~8x | 0.960 |

\* uniformly random high-dim vectors are the worst case for ANN — real
embedding space clusters and lands at 0.95+.
\† extrapolated from scan throughput at that scale.

## Layout

- `src/vector.zig` — SIMD dot product / cosine distance / normalization
- `src/hnsw.zig` — HNSW index (+ connectivity/recall unit tests)
- `src/gemini.zig` — `batchEmbedContents` client (RETRIEVAL_DOCUMENT/QUERY task types)
- `src/tpuf.zig` — turbopuffer v2 REST client (`upsert_columns`, ANN `query`)
- `src/main.zig` — CLI + benchmark harness
