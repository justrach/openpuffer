# Opt-in four-neighbor scoring experiment

This reuses each query SIMD load across four independent int8 dot products. It applies a batching/cache-reuse principle to the existing HNSW walk; it is not a new graph algorithm or a B+ tree port. `Options.batch_neighbor_scoring` defaults to false, is not persisted, and only enables cosine scoring. Other metrics keep scalar scoring. Queue order, filters, graph construction, candidate breadth, reranking, and the existing traversal-prefix policy are unchanged.

## Reproduce

```sh
zig build test
zig build bench-ann-batch -Doptimize=ReleaseFast
```

Benchmarks use macOS arm64, Zig 0.17.0, ReleaseFast, 4,096 seeded synthetic vectors per fixture, 128 queries, and seven alternating scalar/batch pairs of three query passes. CLOCK_UPTIME_RAW measures individual queries. The scalar control is the original scoring path in the same experimental binary, not a separately compiled upstream binary. The benchmark is not installed with the library.

## Paired measurements

Median paired latency reductions, two independent runs (positive means faster):

| Fixture / storage | p50 run 1 / 2 | p95 run 1 / 2 |
| --- | --- | --- |
| clustered-512 / heap | 8.82% / 4.52% | 16.07% / 7.84% |
| clustered-512 / mmap | 7.81% / 8.65% | 0.46% / 18.11% |
| uniform-512 / heap | 3.73% / 6.46% | 2.45% / 4.29% |
| uniform-512 / mmap | 4.23% / 7.67% | 4.51% / 9.32% |
| clustered-1536 / heap | 6.79% / 6.49% | 9.94% / 17.38% |
| clustered-1536 / mmap | 5.51% / 3.60% | 8.77% / 2.39% |

All result IDs and distance bits matched. Sampled current-process RSS was identical in every scalar/batch pair. The kernel adds stack scratch, not persistent allocations; sampled same-process RSS is not a separate-process peak-memory bound.

Recall against exact full-dimensional top-k was unchanged: clustered-512 1.0 and uniform-512 0.568359 (k=24, ef=48, rerank=2); clustered-1536 0.999219 (k=10, ef=128, rerank=4). The low uniform-fixture recall is the existing approximate baseline, not a batching improvement. Upstream's existing 1536D cosine traversal prefix remains unchanged.

## Verification and limits

The final full upstream suite passed on Zig 0.17.0: all 25 build steps succeeded, with 122 executed tests passed and three skipped (other test steps cached). The final targeted batch suite passed all three tests, covering exact dot products and tails, ties, heap/mmap, both f32-storage modes, sequential updates, filters, and scalar fallback for all metrics.

The documented CI development compiler was unavailable from the download server (404). The older minimum compiler failed on existing `@backingInt` calls; Zig 0.17.0 passed instead. No compatibility changes were made outside this experiment.

These are warm synthetic results on one CPU architecture. Real embeddings, cold mmap, larger working sets, independent-process memory measurements, concurrent writers, and x86_64 performance remain unvalidated. Not every fixture consistently clears a 4% threshold, and tails vary noticeably. This is evidence for an opt-in experimental draft, not a production-default change or an end-to-end application speed claim. Adjacent round files contain only benchmark measurements.
