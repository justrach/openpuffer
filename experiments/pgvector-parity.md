# Filters, deletes and metrics: openpuffer next to pgvector

pgvector is the usual way to put vector search next to relational data: rows
carry attributes, queries filter on them, single documents are deleted, and
several distance metrics are available. openpuffer had none of that: cosine
only, no attributes, no filters, no per-document delete. This adds them and
measures the result against pgvector on the same data.

## What was added

| | before | now |
|---|---|---|
| attributes | none | any column besides `id` / `vector` (string, number, bool, null), stored column-wise |
| filters | none | `Eq NotEq In NotIn Lt Lte Gt Gte And Or Not`, `id` as an attribute |
| filtered search | none | few matches: scan only those rows; many: graph walk that accepts only matches (below) |
| delete | whole namespace only | `deletes: [ids]`, `delete_by_filter` (tombstones; the graph stays connected) |
| metrics | cosine | cosine, `euclidean_squared`, `dot_product` (negative inner product) |
| persistence | ids + vectors | WAL entries carry attributes and deletes; snapshots carry metric and attribute columns (envelope v3, written only when needed, so plain cosine namespaces keep v1) |

**Filtered search.** pgvector applies a `WHERE` clause after the index scan:
with `hnsw.ef_search = 40` and a 10% filter, about 4 of the 40 candidates
survive. Its iterative scans (0.8) keep scanning to fill the result, at a
recall cost (below). Here the filter is evaluated first into a bitset over
rows, and the plan depends on how many rows match:

- up to `scanLimit(ef)` ≈ √(n · 32 · ef) matches: scan only those rows. Exact
  up to 2,048 rows; above that, int8 rows plus exact rerank of the best
  `k · rerank_mult`.
- more: walk the graph through every row but admit only matches into the
  result set, until `ef` matches are held. The walk does not stop early because
  rejected rows were close.

Cost model behind the limit: a filtered walk visits about
(unfiltered visits) / (match share) rows, so a scan of `m` matches is cheaper
while m² < visits · n.

## Also fixed: concurrent bulk writes crashed the server

Four clients writing 500-row batches into one namespace killed the server
(SIGABRT in ReleaseFast, a segfault in `ensureInsertRoom` in ReleaseSafe).
Publishing a new row appended to the packed slabs under `splice_mu` alone,
and in-place updates took no lock, while another writer could be growing the
same slabs under the exclusive lock. Publishes and updates now hold the
namespace lock shared, slab growth happens only under the exclusive lock, and
a publisher re-checks for room after taking `splice_mu`. A mixed stress run
(4 writers with upserts, deletes and `delete_by_filter`, 6 readers with
filtered queries, 40 s, ReleaseSafe) answered 29k queries with no crash and no
row that violated its filter or carried another document's attributes.

## Benchmark

100,000 × 768-d vectors derived from real text embeddings (each a mix of two
chunk embeddings plus noise), 500 queries near real queries. Every row has
`scope` (2 values), `kind` (10), `project` (100) and `user_id` (1,000); each
query runs unfiltered and with one equality filter of each selectivity.
Recall is |returned ∩ exact top-10| / 10, so a short result counts against
it. Single client over a kept-alive connection (HTTP for openpuffer, unix
socket for Postgres), p50 latency. macOS arm64, openpuffer ReleaseFast,
PostgreSQL 18.6 + pgvector 0.8.7, HNSW `m = 16, ef_construction = 100` on both.

pgvector has two setups: with btree indexes on the attributes (the planner
then answers selective filters with a bitmap scan and an exact sort) and
without them (filters go through the HNSW index; iterative scans apply).

### ef = 64 (recall / p50)

| filter | pgvector, btree + planner | pgvector HNSW, iterative off | pgvector HNSW, relaxed_order | openpuffer |
|---|---|---|---|---|
| none | 0.84 / 1.31 ms | 0.85 / 1.25 ms | | **0.85 / 0.29 ms** |
| 50% | 0.82 / 1.30 ms | 0.83 / 1.25 ms | 0.83 / 1.28 ms | **0.90 / 0.70 ms** |
| 10% | 1.00 / 23.9 ms (btree + sort) | 0.57 / 1.22 ms, **6.3 of 10 rows** | 0.83 / 2.14 ms | **1.00 / 1.26 ms** |
| 1% | 1.00 / 2.89 ms (btree + sort) | 1.00 / 6.14 ms (seq scan) | same | **1.00 / 0.56 ms** |
| 0.1% | 1.00 / 0.43 ms (btree + sort) | 1.00 / 3.89 ms (seq scan) | same | **1.00 / 0.24 ms** |

With `ef_search = 40` and iterative scans off, pgvector returns 4.1 of 10
rows for the 10% filter (recall 0.38). `strict_order` fills the result
but lands at 0.68–0.74 recall for the 10% filter.

### ef sweep, unfiltered and 50%

| ef | pgvector none | openpuffer none | pgvector 50% | openpuffer 50% |
|---|---|---|---|---|
| 40 | 0.78 / 1.07 ms | 0.78 / 0.29 ms | 0.76 / 1.04 ms | 0.86 / 0.66 ms |
| 64 | 0.84 / 1.31 ms | 0.85 / 0.29 ms | 0.82 / 1.30 ms | 0.90 / 0.70 ms |
| 128 | 0.92 / 2.37 ms | 0.91 / 0.37 ms | 0.95 / 101 ms (see below) | 0.95 / 0.91 ms |

Unfiltered recall at a given ef is the same for both engines: same graph
parameters, same data. The difference is latency.

At ef = 128 the 50% filter sits on the planner's cost boundary: for one
`scope` value it chose the btree bitmap scan plus an exact sort of ~50k rows
(~100 ms), for the other the HNSW scan, so half the queries were slow and
the other half approximate. Without the btree it is 0.91 / 2.1 ms.

### Load and memory

| | pgvector | openpuffer |
|---|---|---|
| load 100k rows | COPY 24 s + `CREATE INDEX` 199 s (4 parallel workers) | 57 s (4 HTTP writers, index built while writing) |
| size | HNSW index 391 MiB, table + indexes 798 MiB | server RSS 462 MiB |

### Unchanged

`bench-synthetic --n 20000 --dim 1536 --ef 128`: same recall, p50 0.68–0.72 ms
before and after (run twice each). Unfiltered queries on a namespace without
deletes take the same code path as before; filtering is a separate
instantiation of the layer search.

## What pgvector still has

Transactions across tables, joins, SQL, point-in-time recovery, halfvec /
bit / sparsevec types, L1 / Hamming / Jaccard, and expression indexes. Array
attributes (`tags Contains x`), full-text ranking, and reclaiming tombstoned
rows (a rebuild after many deletes) are the next gaps here.

## Reproduce

Tests: `zig build test`. Persistence of attributes, deletes and metric across
compaction and restart: `python3 tools/s3_persistence_gate.py` (scenario
`attributes`).
