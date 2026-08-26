# experiments/codedb.md — 512-d clustered code-retrieval profile

This is an **ANN engine** benchmark for the codedb shape. Embedding-model
and GPU time are measured separately and must not be copied here.

Fixed command (do not override):

```
zig build test
zig build -Doptimize=ReleaseFast
./zig-out/bin/openpuffer bench-synthetic \
  --n 20000 --queries 500 --dim 512 --k 24 --ef 128 --rerank-mult 4 --clustered
```

Seed is the harness default (`7`). Driver: `python3 tools/codedb_once.py`.

## Gates

| metric | floor / bar |
|---|---|
| recall@24 | **≥ 0.99** (this deterministic clustered workload) |
| ANN p50 | keep only if ≥4% faster than the same-key best, after a fresh control |
| index construction | discard if > +20% vs same-key best |
| post-query RSS (current MiB) | discard if > +15% vs same-key best |

A result needs **two samples** and a compatible control. `tools/codedb_once.py`
prints `need control pair` until `OPENPUFFER_CONTROL_P50` is set.

Rows are keyed (`key=os-arch-cpu/codedb-20k-512-k24-ef128-rm4`). They never
move the generic 1536-d engine best in `log.md` / `results.svg`.

RSS is **current** resident set (Linux `VmRSS`, macOS `resident_size`). Peak
is printed separately when the OS reports it.

| id | date | p50 (ms) | recall@24 | build_ms | rss_mib | verdict | notes |
|----|------|----------|-----------|----------|---------|---------|-------|
| C000 | 2026-08-26 | — | — | — | — | running | seed row; fill on first same-host pair. Apple M4 Pro control cited in #20 was ~0.094–0.099 ms / recall 1.0 / build 5.5–6.1 s / ~284 MiB. |

## Migration

Do not paste 1536-d engine p50s into this table. Do not paste these p50s
into `experiments/log.md` engine cells.
