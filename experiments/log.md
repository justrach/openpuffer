# experiments/log.md — openpuffer optimization ledger

Metric: `zig build -Doptimize=ReleaseFast && ./zig-out/bin/openpuffer bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128`
Score: lowest **engine-track** `openpuffer (ANN) p50 (ms)` subject to `openpuffer recall@10 >= 0.30`. Gate: `zig build test` (now includes `src/hnsw.zig`). Serve / memory / knobs / scale keeps do not move this score. Run `python3 tools/loop_once.py` after an engine edit.
Protocol: see `program.md`. Append-only EXCEPT completing the current experiment's row; one row per experiment; E000 baseline seeded by human, never edited.
Best-so-far entering E001: p50 = 0.981 ms (the lower of E000's two runs).
Best-so-far on main (M1) after E009: p50 = 0.830 ms (one-ahead qvec prefetch).
Best-so-far on Linux Xeon AVX-512 (parallel cloud-agent lineage, now E012–E022): p50 = 0.575 ms. Not comparable to M1 numbers.
Serving-path (HTTP, Linux, 2026-08-25): p50 1.646 ms → 1.084 ms after E023 → **1.026 ms serial / 0.435 ms keepalive** after E024 (worker pool + inline lone-conn). Not comparable to the in-process ANN metric. Concurrent new-TCP p50 rises on this 4-core host (memory-bound ANN); QPS is the scale score.
Scale (dim=1536, ef=128, 16 GiB host): 1M ANN p50 **12.706 ms** / RSS 11153 MiB (no recall; `--no-exact`). 2M does not fit. Numbers live in the Scale section below — do not put them in the E-table p50 cell or results.svg.
Multi-instance sharding (N serve processes + Python router) is a **separate** design note below the Scale section — not an E-row.

![p50 per experiment](results.svg)

Chart regenerated with `python3 tools/plot_results.py` after every completed experiment.

## SSD-shaped next step (not implemented)

In-memory flatten is the ROI step: Record payload and graph are now contiguous slabs
(`vectors_flat` / `qvecs_flat` / packed CSR). The durable follow-on is a log-structured
map, not a full NVMe/SPDK/ZNS engine:

- **RecordID → {segment, offset, generation}** on large append-only segments.
- Updates append a new record and bump generation; readers ignore stale gen.
- Segments are mmap/O_DIRECT-friendly; compaction rewrites live records.
- Do **not** plumb raw NVMe/SPDK/ZNS until the address map and flatten prove out.

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | track | notes |
|----|------|------------|----------------|---------------|------|----------|-----------|-------------------|---------|-------|-------|
| E000 | 2026-08-23 | baseline (seeded by human, never edit) | none — unmodified tree | — | pass | 1.013 / 0.981 | 0.3130 | — | keep | engine | two consecutive runs; ±3% p50 variance, recall bit-stable; M1 Mac |
| E001 | 2026-08-23 | (mechanics dry-run) comment-only edit produces no perf change | doc-comment line added to src/hnsw.zig | src/hnsw.zig | pass | 0.981 | 0.3130 | 0.0% | discard | engine | out-of-the-box validation of the cycle itself; reverted via git checkout per protocol |
| E002 | 2026-08-23 | widen dotI8 SIMD accumulation 16→64 lanes to cut traversal inner-loop op count | lane=16→64 in dotI8, 4 independent i32 accumulators, integer math unchanged (bit-exact) | src/vector.zig | pass | 1.026 / 1.015 | 0.3130 | +4.6% / +3.5% | discard | engine | post-hoc control: reverted tree under identical load read 1.008 → variant was NEUTRAL not slower; ambient ~2-3% drift vs the 0.981 record (peer building concurrently); reverted |
| E003 | 2026-08-23 | dotI8: multiply in i16 (exact: max product 16129 < 32767), widen only the product to i32 | lane stays 64; operand casts target i16 not i32; product widened once | src/vector.zig | pass | 1.026 / 1.007 | 0.3130 | +4.6% / +2.7% | discard | engine | neutral vs same-load control (1.008); lane-16 lowering already near-optimal on M1 NEON; reverted |
| E004 | 2026-08-23 | flatten int8 quantized pool into one contiguous buffer (traversal currently hops across 20k separate heap allocations) | qvecs: ArrayList([]i8) → qvecs_flat: ArrayList(i8), stride dim; f32 pool left per-vector (server.zig snapshot reads it) | src/hnsw.zig | pass | 0.951 / 0.957 | 0.3130 | −3.1% / −2.4% | discard | engine | REAL but sub-threshold: both runs beat 0.981 best and agree tightly (~3% win), yet <4% keep bar — reverted per protocol. Strong candidate to re-land STACKED with another orthogonal optimization as one compound hypothesis |
| E005 | 2026-08-23 | COMPOUND: flat int8 pool (E004, proven ~3%) + persistent query scratch — visited bitset reused across queries, manual array heaps replace per-call PriorityQueues (≈10 malloc/free per query eliminated) | qvecs_flat stride-dim pool; QueryScratch{visited,cand,res,out} grown once on Self; searchLayer fills scratch, callers read sc.out; public API unchanged | src/hnsw.zig | pass | 1.001 / 0.982 | 0.3120 | +2.0% / +0.1% | discard | engine | NO net win — implies E004's ~3% was mostly ambient drift, not the flat pool; alloc churn is noise at this scale. Found+fixed a real mixed-allocator bug on the way (scratch grown under caller's arena, freed under index's gpa → invalid free at deinit); scratch must always use self.allocator if ever re-landed. Recall 0.3120 vs 0.3130 = heap tie-break order differs from PriorityQueue, harmless |
| E006 | 2026-08-23 | quantized traversal needs only candidate ordering before exact reranking, so per-query scale and the f64 conversion, `1 -`, and nonnegative clamp are unnecessary in every node comparison | used monotonic f32 score `-(dotI8 * stored_qscale)`; removed `qs` from distQ/QDist and eliminated the second query amax scan; manually reverted byte-identically after measurement could not run | src/hnsw.zig | pass | — | — | — | discard | engine | `zig build test` and ReleaseFast build passed; both exact benchmark invocation attempts were blocked by the command-execution policy before launch, so no qualifying measurement exists. Chart regeneration was also blocked; the SVG remains unchanged because rows without numeric p50 are skipped. Pre-existing changes in src/main.zig, src/server.zig, dashboard.html, src/persist.zig, src/r2probe.zig, src/s3.zig, and src/uriprobe.zig preserved |
| E007 | 2026-08-23 | memory-latency hiding: 20k × 1536-dim int8 pool ≈ 30MB exceeds LLC, so each unvisited neighbor stalls on a main-memory fetch — software-prefetch the NEXT neighbor's vector while evaluating the current one | @prefetch(qvecs.items[next]) issued one-ahead in the neighbor loop of searchLayer | src/hnsw.zig | pass | 0.949 / 0.990 | 0.3130 | −3.3% / +0.9% | discard | engine | NOT reproducible; same-load control of the REVERTED tree read 0.915/0.950 — ambient drift (±4-5%) now exceeds every effect measured since E002, and the 0.981 reference was stale-high. Protocol amended: periodic re-baselining (see program.md). Prefetch may still help under quieter conditions |
| E008 | 2026-08-23 | graph quality: ef_construction 100→300 builds a better-connected index so queries need fewer hops → fewer distance evals per query; build time is NOT part of the scored metric (cost noted honestly) | Options.ef_construction default 100→300 in src/hnsw.zig; bench builds with `.{}` defaults via main.zig:231 (unmodified) | src/hnsw.zig | pass | 0.962 / 0.955 | 0.3175 | +2.8% / +2.0% | discard | engine | recall ROSE 0.3130→0.3175 (graph genuinely better, bit-stable) but p50 did NOT follow: at fixed ef=128 denser neighbor lists add more per-hop evals than fewer hops save — net neutral-to-slightly-worse vs fresh control 0.936/0.955. Consistent with meta-lesson: traversal work is pinned by ef, not graph degree. Reverted; build-time cost also unattractive (~3x insert cost) |
| E009 | 2026-08-23 | RETRY of E007 under quiet conditions: prefetch the NEXT neighbor's int8 vector one-ahead while evaluating the current one (30MB pool > LLC ⇒ each unvisited neighbor risks a main-memory stall; today's control runs agree within 2% vs E007's ±5% chaos) | @prefetch(qvecs.items[neighbors[ni+1]]) issued in the neighbor loop of searchLayer before dist_ctx.dist(nid) | src/hnsw.zig | pass | 0.830 | 0.3130 | −11.3% vs fresh control (0.936); −15.4% vs stale historical best (0.981) | keep | engine | FIRST kept improvement of the loop. Run 1 cleared the ≥4% bar outright so no Run 2 needed. Recall bit-stable ⇒ identical traversal order, pure latency win. Confirms E007 hypothesis was correct but unmeasurable amid ambient drift; quiet machine (control pair within 2%) finally resolves it. New best: 0.830 ms |
| E010 | 2026-08-23 | deepen memory-level parallelism: one-ahead prefetch barely covers ~100ns memory latency against a ~150ns dot, so issue TWO ahead and also prefetch qscales[next] (separate array, random access, needed right after each dot) | prefetch loop in searchLayer: qvecs[neighbors[ni+1]] AND qvecs[neighbors[ni+2]] AND qscales[neighbors[ni+1]] | src/hnsw.zig | pass | 0.868 / 0.864 | 0.3130 | +3.7% / +3.2% vs fresh control (0.841/0.837) | discard | engine | consistent pair, real regression: deeper prefetch over-fetches — lines for visited-skipped neighbors and far-ahead loads evict useful lines before use. One-ahead (E009) is the sweet spot for this access pattern. Reverted to E009 state |
| E011 | 2026-08-23 | COMPOUND micro-overhead: (a) re-land E006 — traversal only ORDERS candidates, so distQ can use monotonic f32 −(dot·scale_id), skipping per-eval f64 widen/multiply, f64→f32 narrow, `1−`, clamp; (b) hoist the results-heap worst-distance peek() out of the neighbor inner loop into a local updated only on push/pop | distQ body simplified (qs param now unused, kept for API stability); searchLayer neighbor loop caches `w`, refreshes after pop or when d becomes new max | src/hnsw.zig | pass | 0.847 / 0.943 | 0.3130 | +1.2% / +12.2% vs control (0.841/0.837) | discard | engine | no win — run 1 within noise of control (bookkeeping was already sub-noise), run 2 drifted high (ambient). The eval inner loop is now prefetch-fed (E009) and the dot dominates; scalar overhead around it is unmeasurable at this machine's noise floor. Reverted to E009 state. Engine plateau: ~0.83-0.85ms at ef=128 on this hardware |
| E012 | 2026-08-23 | rebaseline unmodified tree on Linux Xeon AVX-512 (this agent host); M1 numbers are not comparable | none — no source edit | — | pass | 1.031 / 0.927 | 0.3130 | — | keep | engine | parallel cloud-agent row (originally E004). 4-core Intel Xeon (KVM), avx512f/vnni/amx; first-run p99=5.08ms (cold); host variance > M1 ±3%. Best for this host = 0.927 |
| E013 | 2026-08-23 | widen dotI8 16→64 so the i8 inner loop fills a 512-bit AVX-512 register (E002 was neutral on M1 NEON) | lane=16→64 in dotI8; integer math unchanged (bit-exact) | src/vector.zig | pass | 0.875 / 0.834 | 0.3130 | -5.6% / -10.0% | keep | engine | parallel cloud-agent row (originally E005). both runs ≥4% vs host 0.927; recall bit-stable; new host best 0.834 |
| E014 | 2026-08-23 | stack-allocate 1536-dim query/quantize scratch (norm_buf/qbuf are 512, so every metric query heap-allocs) | grow searchAdvanced scratch from 512 to 2048 | src/hnsw.zig | pass | 0.774 / 0.845 | 0.3130 | -7.2% / +1.3% | discard | engine | parallel cloud-agent row (originally E006). run1 looked like a keep but exact-scan also dropped 5.32→4.79 (ambient); run2 0.845 = neutral vs 0.834. Reverted |
| E015 | 2026-08-23 | prefetch next neighbor's int8 vector during layer traversal to hide random qvec load latency | @prefetch next qvec in searchLayer neighbor loop | src/hnsw.zig | pass | 0.616 / 0.678 | 0.3130 | -26.1% / -18.7% | keep | engine | parallel cloud-agent row (originally E007). both runs clear 4%; ANN/exact ratio 0.128 and 0.123 vs ~0.157 baseline — not ambient. New host best 0.616 |
| E016 | 2026-08-23 | reuse candidate/result heaps across searchLayer calls instead of allocating two PriorityQueues per layer | scratch heaps owned by searchAdvanced, reset per layer | src/hnsw.zig | pass | 0.615 / 0.650 | 0.3130 | -0.2% / +5.5% | discard | engine | parallel cloud-agent row (originally E008). arena-backed bench alloc already makes queue alloc cheap; no 4% win. Reverted |
| E017 | 2026-08-23 | 64-byte-align quantized vectors so AVX-512 i8 loads are aligned | alignedAlloc(i8, 64) on insert + align(64) query qbuf | src/hnsw.zig | pass | 0.687 / 0.610 | 0.3130 | +11.5% / -1.0% | discard | engine | parallel cloud-agent row (originally E009). unaligned AVX-512 loads already cheap; no 4% win. Reverted |
| E018 | 2026-08-23 | prefetch 8 cache lines of the next qvec (1536 B = 24 lines; E015 only touched the first) | 8×64 B @prefetch of next neighbor qvec | src/hnsw.zig | pass | 0.662 / 0.717 | 0.3130 | +7.5% / +16.4% | discard | engine | parallel cloud-agent row (originally E010). extra prefetches slower — cache pollution / frontend cost. Reverted |
| E019 | 2026-08-23 | two independent 64-lane i8 accumulators to hide i32 add latency (bit-exact) | dotI8 dual-acc, 128 i8s per iteration | src/vector.zig | pass | 0.740 / 0.699 | 0.3130 | +20.1% / +13.5% | discard | engine | parallel cloud-agent row (originally E011). dual 64-wide i32 accs slower (register pressure). Reverted |
| E020 | 2026-08-23 | prefetch next f32 vector during exact rerank (40 random 6 KiB loads) | @prefetch next stored vector in rerank loop | src/hnsw.zig | pass | 0.678 / 0.630 | 0.3130 | +10.1% / +2.3% | discard | engine | parallel cloud-agent row (originally E012). rerank is a small slice of p50; prefetch did not clear 4%. Reverted |
| E021 | 2026-08-23 | two independent 16-wide f32 accumulators in dot() to hide FMA latency on rerank | dual-acc f32 dot, bit-exact enough for ranking | src/vector.zig | pass | 0.674 / 0.610 | 0.3130 | +9.4% / -1.0% | discard | engine | parallel cloud-agent row (originally E013). query p50 not a 4% win (build *was* faster, ~38.7s vs ~46s, but metric is query). Reverted |
| E022 | 2026-08-23 | only prefetch the next neighbor qvec when it is not already visited (skip wasted prefetches) | guard E015 prefetch with !visited.isSet | src/hnsw.zig | pass | 0.710 / 0.575 | 0.3130 | +15.3% / -6.7% | keep | engine | parallel cloud-agent row (originally E014). run1 miss, run2 decides (-6.7%); recall 0.3130. exact-scan also dipped on run2 (5.55→4.85) so treat 0.575 as noisy; still a protocol keep. New host best 0.575 |
| E023 | 2026-08-25 | HTTP serve overhead (not ANN dots) is the leftover ~1ms on Linux; io_uring accept/recv/send + TCP_NODELAY + scan query JSON without a 1536-node AST | Linux io_uring serve loop; nodelay; parseAnnQuery; tools/serve_bench.py | src/iouring_sock.zig src/server.zig | pass | 1.084 / 1.089 / 1.113 | — | −34% vs HTTP baseline 1.646 | keep | serve | SERVING metric (urllib, new TCP conn/query, n=2000 dim=1536). Engine bench-synthetic unchanged. Chart not regenerated — different metric. |
| E024 | 2026-08-25 | at-scale HTTP: E023 serialized all connections on one accept+handle loop; a fixed OS-thread pool should run concurrent ANN queries on all cores and keep-alive without per-conn pthread_create | io_uring accept + ncpu posix keep-alive workers; per-worker arena; inline lone-conn (drain backlog to pool); serve_bench concurrent keepalive | src/server.zig src/iouring_sock.zig tools/serve_bench.py | pass | HTTP serial 1.026 / ka 0.435 | — | −5.4% vs E023 serial 1.084; ka −60% | keep | serve | SERVING scale (not ANN chart). n=2000 dim=1536: conc8 6.845ms / 988 qps, conc32 19.7ms / 1065 qps, ka+conc8 1.483ms / 1298 qps. n=20000: serial 1.931ms, ka 1.292ms. /proc: all 4 workers take CPU — conc p50 is 4-core DRAM-bound, not a 1-thread queue. |
| E025 | 2026-08-25 | 1M-scale RSS (~9 GB at 800k, 11.2 GB at 1M) is mostly GPA headers + size-class waste on ~4M live allocs, not payload; flatten f32/i8 pools and pack the graph as fixed-slack CSR so the layout is mmap/SSD-shaped | vectors_flat + qvecs_flat stride dim; layer-0/higher-layer packed neighbors (stride m0+1 / m+1); accessors for distTo/distQ/prefetch/update/planInsert/commitInsert/serialize/load | src/hnsw.zig | pass | 0.744 / 0.581 | 0.3130 | +29% / +1.0% vs host best 0.575 (run2 within noise) | keep | memory | MEMORY keep (not a p50 keep — E004's ~3% was below the 4% bar; re-landed because 1M RSS is the score). Clustered recall@10 = 1.0; serialize/load test pass. RSS no-corpus GPA: n=20k 221.1→150.2 MiB (−32%); n=50k 527.0→374.3 MiB (−29%). Old 1M bench (left running, finished on its own) post-build 11153 MiB / p50 12.7ms. Linear 1M estimate for flatten ≈ 7.5 GiB. Chart not regenerated (E023/E024 serving cells stay HTTP-tagged; do not mix into results.svg). No NVMe/SPDK/ZNS. |
| E026 | 2026-08-25 | Pareto surface at scale is an (n, ef, distribution) problem, not a 20k p50 micro-opt; expose the existing rerank_mult knob and one-build ef/clustered sweeps so we can find the smallest ef that holds recall@10 ≥ 0.30 | CLI `--rerank-mult` / `--ef-sweep` / `--clustered` (mixture-of-Gaussians); serve + per-query `rerank_mult`; search() default still 4 | src/hnsw.zig src/main.zig src/server.zig tools/qa_bench.py | pass | 0.639 | 0.3130 | +11% vs E025 run2 0.575 (within flatten-host noise; not a p50 claim) | keep | knobs | KNOB/DOCS keep. Scored metric unchanged (searchAdvanced(..., 4) ≡ search()). Scale n×ef lives in the Pareto section below — do not read 1M/200k p50s from this cell. Random cannot hold 0.30 at 200k even at ef=1024; clustered holds 0.98–1.00 from ef=64 through 200k. Flatten 1M `--no-exact`: RSS **7481 MiB** / ef=128 p50 **1.410 ms** (old layout 11153 MiB / 12.7 ms). |
| E027 | 2026-08-25 | drop the hot-path f32 copy (optional store_f32 / `--no-f32`) so 2M×1536 can fit on 16 GiB after E025 flatten | Options.store_f32 default true; `--no-f32` skips f32 slab; int8 traversal; rerank only if f32 present; serialize v2 omits f32, load still accepts v1 | src/hnsw.zig src/main.zig | pass | 0.679 | 0.3105 | +18% vs E022 0.575 (not a p50 claim) | keep | memory | MEMORY keep. Scored command unchanged (default still stores f32). `--no-f32` 20k recall **0.3100** (−0.0005). RSS `--no-exact`: 50k 375→**82** MiB; 200k 1689→**323** MiB. 2M `--no-exact --no-f32 --ef 128` **fits**: post-build **3222 MiB** (p50 1.760 ms — Scale section only, not this cell / not results.svg). Flatten 1M f32+i8 was 7481 MiB; dropping f32 is the ~5.7 GiB save. |

## Scale (dim=1536, ef=128) — 2026-08-25, 16 GiB / 4-core Xeon, no swap

Not an E-row. Numeric ANN p50s here must not be copied into the table above or `results.svg` (they are 10–20× the 20k metric and would flatten the chart). HTTP p50 stays out of the chart.

Host: MemTotal 16398384 kB, ulimit -v 14 GiB so a failed alloc dies instead of taking the machine. Payload estimate: n × 1536 × 5 B = 7.7 GiB (1M) / 15.4 GiB (2M) plus HNSW graph and GPA overhead.

Exact commands:

```
./zig-out/bin/openpuffer bench-synthetic --n 1000000 --queries 50 --dim 1536 --k 10 --ef 128 --no-exact
./zig-out/bin/openpuffer bench-synthetic --n 200000 --queries 50 --dim 1536 --k 10 --ef 128
./zig-out/bin/openpuffer bench-synthetic --n 50000 --queries 50 --dim 1536 --k 10 --ef 128
./zig-out/bin/openpuffer bench-synthetic --n 20000 --queries 50 --dim 1536 --k 10 --ef 128
python3 tools/serve_bench.py --n 20000 --queries 50 --dim 1536 --k 10 --ef 128 --keepalive --port 8098
```

2M with f32 stored was **not launched** on the flatten tree (2×7481 ≈ 14.6 GiB). E027 `--no-f32` opened it: see the Drop-f32 RSS subsection below.

| n | build | ANN p50 | p95 | p99 | recall@10 | RSS | notes |
|---|-------|---------|-----|-----|-----------|-----|-------|
| 20k | 43.8 s (457 vec/s) | 0.532 ms | 1.596 ms | 3.820 ms | **0.3280** | — | control; 50 queries (metric uses 200 → recall 0.313) |
| 50k | 119.1 s (420 vec/s) | 1.237 ms | 4.901 ms | 8.579 ms | **0.1860** | — | recall already under the 0.30 floor |
| 200k | 661.4 s (302 vec/s) | 2.826 ms | 8.295 ms | 11.085 ms | **0.0400** | — | exact scan p50 107 ms; random 1536-d collapse |
| 1M | 4657.7 s (215 vec/s) | 12.706 ms | 17.359 ms | 23.969 ms | skipped (`--no-exact`) | **11153 MiB** post-build / 11160 post-query | mean 12.645 ms, 3954 qps |
| 2M | — | — | — | — | — | would be ~22 GiB old-layout / ~15 GiB flatten+f32 | skipped on f32-on trees |
| 2M `--no-f32` | 3767 s (531 vec/s) | 1.760 ms | 2.298 ms | 2.298 ms | skipped (`--no-exact`) | **3222 / 3227 MiB** | E027; **fits 16 GiB**. Do not copy this p50 into the E-table or results.svg. |
| HTTP 20k ka | upsert 60.0 s | 1.118 ms | 2.445 ms | 3.997 ms | nonempty 50/50 | — | `serve --ef 128`; E024 ka at n=20k was 1.292 ms (old implicit ef=256) |

Recall@10 on uniform random 1536-d vectors falls off a cliff as n grows at fixed ef=128 (0.33 → 0.19 → 0.04). The 1M p50 is a latency number, not a quality number. Clustered data (qa_bench) historically held recall ≈ 1.0 at ef=256; qa_bench now **pins `--ef 256`** because serve default changed 256→128.

What is left at 1–2M:

- **Recall at scale**: raise ef (or M / ef_construction) once n is large; measure on clustered embeddings, not only random.
- **Memory layout**: E025 flatten **measured** at 1M `--no-exact`: RSS **7481 MiB** (old layout 11153 MiB, −33%). Payload was the estimate; GPA waste is gone.
- **Drop the f32 copy** (E027, done): `--no-f32` / `store_f32=false`. 50k 375→82 MiB; 200k 1689→323 MiB; **2M fits at 3222 MiB**. Default still stores f32 so the 20k scored bench is unchanged.
- **mmap** the flat pools so RSS is demand-paged.
- `rerank_mult` is now `--rerank-mult` (default still 4). At 20k/ef=128, mult=1 and mult=8 both recall **0.3280** — rerank cannot recover neighbors that ef never found.
- Do not HTTP-upsert 1–2M; in-process insert is the only sane load path.

## Drop-f32 RSS (E027) — 2026-08-25, same 16 GiB host

Not an E-row. Do not copy the 2M p50 into the ledger table or `results.svg`.

Default `store_f32=true`. `--no-f32` skips the packed f32 slab; search is int8-only (no exact rerank). `--no-exact` so RSS is the index, not a second corpus.

| n | store_f32 | post-build RSS | saved | notes |
|---|-----------|----------------|-------|-------|
| 20k (corpus retained) | on | 285 MiB | | scored path; recall 0.3105 |
| 20k (corpus retained) | off | 151 MiB | 134 MiB | recall **0.3100** (−0.0005) |
| 50k `--no-exact` | on | **375 MiB** | | matches E025 flatten 374.3 |
| 50k `--no-exact` | off | **82 MiB** | 293 MiB | = n×1536×4 |
| 200k `--no-exact` | on | **1689 MiB** | | |
| 200k `--no-exact` | off | **323 MiB** | 1366 MiB | |
| 2M `--no-exact --ef 128` | off | **3222 / 3227 MiB** | ~12 GiB vs flatten+f32 | **fits**; ANN p50 1.760 ms (20 queries; latency-only) |
| 2M | on | ~17 GiB (200k×10) | | skipped; over 14 GiB bar |

20k `--no-f32` recall stays above the 0.30 floor. On this random 1536-d set rerank was not finding extra neighbors (E026 `rerank_mult` same lesson).

## Pareto (n × ef, dim=1536) — 2026-08-25, flatten tree, 50 queries

Not an E-row. Do not copy these p50s into the ledger table or `results.svg`.

Scored metric on this tree (200 queries, ef=128, random): **p50 0.639 ms**, recall@10 **0.3130**, RSS 297 / 335 MiB. Neutral vs E025 (0.744 / 0.581); harness-only.

`--clustered` is a mixture of `max(32, n/256)` unit centers + noise 0.35 (cosine to center ≈ 0.94). Live `qa_bench.py` was **not** run (no GEMINI/TPUF keys). Clustered is the product-relevant quality axis; uniform random 1536-d is the known cliff.

### Random (uniform Gaussian) — recall cliff

| n | ef | p50 (ms) | p95 (ms) | recall@10 | RSS post-build / post-query | holds ≥0.30? |
|---|-----|----------|----------|-----------|-----------------------------|--------------|
| 20k | 64 | 0.442 | 0.621 | 0.1900 | 297 / 306 MiB | no |
| 20k | 128 | 0.618 | 0.703 | **0.3280** | 297 / 315 MiB | **yes** (smallest measured) |
| 20k | 256 | 0.930 | 0.978 | 0.4960 | 297 / 325 MiB | yes |
| 20k | 512 | 1.473 | 1.514 | 0.7160 | 297 / 335 MiB | yes |
| 20k | 1024 | 2.244 | 2.289 | 0.9140 | 297 / 346 MiB | yes |
| 50k | 64 | 0.758 | 0.883 | 0.1220 | 731 / 752 MiB | no |
| 50k | 128 | 1.273 | 1.454 | 0.1860 | 731 / 773 MiB | no |
| 50k | 256 | 2.171 | 2.392 | 0.2940 | 731 / 794 MiB | no (just under) |
| 50k | 512 | 3.446 | 3.923 | **0.4500** | 731 / 816 MiB | **yes** (smallest measured) |
| 50k | 1024 | 4.814 | 5.210 | 0.6580 | 731 / 839 MiB | yes |
| 200k | 64 | 1.249 | 1.505 | 0.0220 | 2733 / 2812 MiB | no |
| 200k | 128 | 1.692 | 1.950 | 0.0400 | 2733 / 2891 MiB | no |
| 200k | 256 | 2.940 | 3.175 | 0.0760 | 2733 / 2970 MiB | no |
| 200k | 512 | 5.043 | 5.477 | 0.1420 | 2733 / 3050 MiB | no |
| 200k | 1024 | 8.480 | 8.865 | 0.2640 | 2733 / 3132 MiB | no (still under; random cannot hold 0.30) |
| 1M | 64 | 0.857 | 1.022 | skipped (`--no-exact`) | **7481 / 7488 MiB** | latency/RSS only |
| 1M | 128 | 1.410 | 1.646 | skipped | 7481 / 7495 MiB | flatten vs old-layout 12.706 ms / 11153 MiB |
| 1M | 256 | 2.443 | 2.842 | skipped | 7481 / 7503 MiB | |
| 1M | 512 | 4.446 | 4.929 | skipped | 7481 / 7511 MiB | |
| 1M | 1024 | 8.583 | 9.219 | skipped | 7481 / 7520 MiB | one run; build 3970 s (252 vec/s) |

20k rerank_mult at ef=128 (random, 50q): mult=1 p50 0.551 ms recall 0.3280; default 4 p50 0.618 ms recall 0.3280; mult=8 p50 0.591 ms recall 0.3280. Same recall — not a quality knob on this cliff.

**Random cannot hold recall@10 ≥ 0.30 at n=200k** even at ef=1024 (0.264). Doubling ef roughly doubles recall on this cliff (0.022 → 0.040 → 0.076 → 0.142 → 0.264); 0.30 would need ~ef=1400 and is the wrong axis. Treat clustered as the quality number.

### Clustered (SQuAD-style mixture)

| n | ef | p50 (ms) | p95 (ms) | recall@10 | RSS post-build / post-query | holds ≥0.30? |
|---|-----|----------|----------|-----------|-----------------------------|--------------|
| 20k | 64 | 0.160 | 0.174 | **1.0000** | 268 / 278 MiB | **yes** (smallest measured; 78 centers) |
| 20k | 128 | 0.189 | 0.201 | 1.0000 | 268 / 287 MiB | yes |
| 20k | 256 | 0.240 | 0.341 | 1.0000 | 268 / 296 MiB | yes |
| 20k | 512 | 0.524 | 0.677 | 1.0000 | 268 / 305 MiB | yes |
| 20k | 1024 | 0.804 | 0.983 | 1.0000 | 268 / 316 MiB | yes |
| 50k | 64 | 0.204 | 0.218 | **1.0000** | 670 / 691 MiB | **yes** (195 centers) |
| 50k | 128 | 0.249 | 0.262 | 1.0000 | 670 / 711 MiB | yes |
| 50k | 256 | 0.327 | 0.398 | 1.0000 | 670 / 732 MiB | yes |
| 50k | 512 | 0.782 | 1.086 | 1.0000 | 670 / 753 MiB | yes |
| 50k | 1024 | 1.480 | 1.844 | 1.0000 | 670 / 775 MiB | yes |
| 200k | 64 | 0.268 | 0.370 | **0.9800** | 2675 / 2753 MiB | **yes** (781 centers) |
| 200k | 128 | 0.314 | 0.477 | 0.9980 | 2675 / 2832 MiB | yes |
| 200k | 256 | 0.478 | 0.640 | 0.9980 | 2675 / 2911 MiB | yes |
| 200k | 512 | 1.101 | 1.488 | 0.9980 | 2675 / 2991 MiB | yes |
| 200k | 1024 | 2.100 | 2.473 | 0.9980 | 2675 / 3071 MiB | yes |

### Recommended operating points (so far)

| n | random (quality floor 0.30) | clustered (product) | notes |
|---|----------------------------|---------------------|-------|
| 20k | **ef=128** (0.33 @ 0.62 ms) | **ef=64** (1.00 @ 0.16 ms) | scored default is already the random floor |
| 50k | **ef=512** (0.45 @ 3.45 ms); ef=256 is 0.294 | **ef=64** (1.00 @ 0.20 ms) | random cannot hold 0.30 at serve default 128 |
| 200k | **cannot hold 0.30** (ef=1024 → 0.264 @ 8.48 ms) | **ef=64** (0.98 @ 0.27 ms) or ef=128 (0.998 @ 0.31 ms) | random is not the quality axis at this n |
| 1M | latency/RSS only (ef=128 → **1.410 ms**, RSS **7481 MiB**) | treat as clustered: **ef=64–128** | flatten vs old 12.7 ms / 11153 MiB; no recall (would not fit corpus+index) |

Serve default **ef=128** is the right product default: clustered holds ≥0.98 at 20k–200k, and 1M query p50 stays 1.4 ms. Raise `--ef` only for uniform-random-like workloads (20k: 128, 50k: 512, 200k: cannot hold 0.30).

1M `--no-exact` p50s are the clean latency numbers (no brute-force cache pollution). 20k–200k rows above compute exact recall after each query, which dirties cache — treat those p50s as slightly pessimistic vs a recall-free serve path. 200k random p50 is also inflated by a 1.2 GiB exact scan between queries.

## Multi-instance sharding (not an E-row) — 2026-08-25

Scale path that does **not** put 2M vectors in one 16 GiB address space:
N independent `openpuffer serve` processes + a thin Python router
(`tools/shard_router.py`). Chart / E-table p50 cells are unchanged.

```
python3 tools/shard_router.py --shards 4 --port 8800 --ef 128
OPENPUFFER_SHARDS=4 python3 tools/shard_router.py --port 8800
python3 tools/shard_bench.py --n 8000 --queries 80 --dim 1536 --ef 128
```

### Shard key

FNV-1a 64-bit (`tools/shard_key.py`). Canonical bytes for the default
`doc` mode:

```
utf-8(namespace) + 0x00 + utf-8(decimal_id)     then  key % N
```

Integer ids are ASCII decimal with no leading zeros, so `42` and `"42"`
hash the same. Namespace mode hashes only `utf-8(namespace)` — a tenant
fits on one process; that does **not** split a 2M-doc namespace.

Writes (upsert) split the batch and send each document to exactly one
child. Queries scatter-gather to all children (doc mode) or the owning
child (namespace mode), then merge rows by `$distance` ascending.

### Measured (this host, 4-core Xeon, 16 GiB, flatten+ef=128 tree)

n=8000 × 1536, ef=128, k=10, 80 queries, per-child `--workers 2`.
Router uses a **new localhost TCP** to each child per request (honest
first-experiment hop; keep-alive reuse against serve reset under
scatter-gather). Same total vectors in every row.

| layout | shards | upsert | ka p50 | ka QPS | ka+c4 p50 | ka+c4 QPS | child RSS | route |
|---|---|---|---|---|---|---|---|---|
| 1-direct (no router) | 1 | 20.5 s | **0.749 ms** | 787 | 0.814 ms | 966 | 86.2 MiB | n/a |
| router-1 | 1 | 27.4 s | **1.208 ms** | 558 | 4.792 ms | 696 | 86.2 MiB | n/a |
| router-2 | 2 | 17.9 s | **1.455 ms** | 495 | 6.402 ms | 565 | 88.1 MiB | 32/32 ok |
| router-4 | 4 | 14.1 s | **2.113 ms** | 378 | 9.012 ms | 417 | 92.5 MiB | 32/32 ok |

Smoke n=200 was the same shape (direct 0.48 ms, router-1 1.21 ms,
router-4 1.61 ms). Clustered merge (400 docs, 4 centroids, k=10):
**router-4 top-k == 1-direct top-k == exact**, overlap 1.000.

Write routing: sampled docs appear only on `shard_for(ns, id, N)`.
GET count across children sums to n.

### What this means for 1M / 2M

Do **not** read the 8k p50s as “sharding is slower at 1M”. At 8k the
graph walk is ~0.7 ms; the Python hop (~0.46 ms router-1 − direct) and
N parallel child requests dominate. At 1M the old-layout walk was 12.7 ms; flatten measured **1.41 ms /
7481 MiB** (see Pareto). Same-box 4×250k is still not a RAM win.

| | 1×1M | 4×250k |
|---|---|---|
| payload | ~7.5 GiB flatten | same bytes, split ~1.9 GiB/process |
| process RSS | one address space | 4 heaps + 4 copies of code; at 8k the extra was only +6 MiB |
| fits 16 GiB? | yes after flatten (~7.5 vs old 11.2) | yes, each child well under ulimit |
| 2M | ~15 GiB + overhead — **does not fit one process** | 4×500k ≈ 3.8 GiB/process. Total RAM still ~15 GiB so one 16 GiB box is tight; **four boxes (or a bigger one) is the actual 2M path** |
| query p50 | 12.7 ms (old 1M) | expect ~max(per-shard walk) + hop + merge. 200k in-process was 2.8 ms; 4-way DRAM contention on this 4-core host will eat some of that |
| recall@10 random ef=128 | 0.04 at 200k, unknown at 1M | each shard is a smaller graph (250k ≈ 0.04? or better than 1M). Merge of per-shard ANN ≠ one 1M walk |
| QPS | one walk per query | N walks per query — **hurts** concurrent QPS unless you have N machines |

Sharding is an **address-space / isolation / multi-box** move, not a
free latency win on one 4-core box at modest n.

### Honest limits

1. **Cross-shard recall.** Merge is of *per-shard HNSW* top-k, not a
   unified graph walk. If a shard’s local ANN misses a true neighbor
   that lives on that shard, the merge misses it. On clustered data at
   n=400 this did not show up (overlap 1.0). On uniform random 1536-d
   at large n, per-shard recall is already the story (see Scale table).
2. **Network hop.** Every query is client → Python router → N child
   HTTP requests → merge. Measured hop ≈ **0.46 ms** (router-1 −
   1-direct) with a new TCP per child. A Zig/io_uring router with
   keep-alive would shrink this; it would not remove scatter-gather.
3. **Merge cost** is tiny (N×k sort). The p50 rise 1.21 → 2.11 ms from
   1→4 shards is waiting on N child searches + N handshakes, not the
   sort.
4. **Oversubscription.** 4 shards × `--workers 2` = 8 serve threads on
   4 cores, plus the router. Concurrent QPS fell as N grew. Pin
   `--workers 1` per child on a 4-core host if you care about one
   query’s DRAM, or put shards on separate machines.
5. **Keep-alive to children.** First attempt reused HTTP connections
   and reset under scatter-gather (serve’s lone-conn path pins accept
   when workers=1; throwaway executor threads reset sockets). Fresh
   connections are the working experiment; pooling is follow-up.
6. **Not implemented:** NVMe/SPDK, Zig router, resharding, replication,
   cross-process snapshot. Persistence still lives inside each child.

Verdict: the path works. Use it when one process cannot hold the
vectors (2M, or 1M + room for other things). Do not turn it on at 8k
expecting a p50 win.
