# experiments/log.md — openpuffer optimization ledger

Metric: `zig build -Doptimize=ReleaseFast && ./zig-out/bin/openpuffer bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128`
Score: lowest `openpuffer (ANN) p50 (ms)` subject to `openpuffer recall@10 >= 0.30`. Gate: `zig build test` must pass.
Protocol: see `program.md`. Append-only EXCEPT completing the current experiment's row; one row per experiment; E000 baseline seeded by human, never edited.
Best-so-far entering E001: p50 = 0.981 ms (the lower of E000's two runs).

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | notes |
|----|------|------------|----------------|---------------|------|----------|-----------|-------------------|---------|-------|
| E000 | 2026-08-23 | baseline (seeded by human, never edit) | none — unmodified tree | — | pass | 1.013 / 0.981 | 0.3130 | — | keep | two consecutive runs; ±3% p50 variance, recall bit-stable; M1 Mac |
| E001 | 2026-08-23 | (mechanics dry-run) comment-only edit produces no perf change | doc-comment line added to src/hnsw.zig | src/hnsw.zig | pass | 0.981 | 0.3130 | 0.0% | discard | out-of-the-box validation of the cycle itself; reverted via git checkout per protocol |
| E002 | 2026-08-23 | widen dotI8 SIMD accumulation 16→64 lanes to cut traversal inner-loop op count | lane=16→64 in dotI8, 4 independent i32 accumulators, integer math unchanged (bit-exact) | src/vector.zig | pass | 1.026 / 1.015 | 0.3130 | +4.6% / +3.5% | discard | post-hoc control: reverted tree under identical load read 1.008 → variant was NEUTRAL not slower; ambient ~2-3% drift vs the 0.981 record (peer building concurrently); reverted |
| E003 | 2026-08-23 | dotI8: multiply in i16 (exact: max product 16129 < 32767), widen only the product to i32 | lane stays 64; operand casts target i16 not i32; product widened once | src/vector.zig | pass | 1.026 / 1.007 | 0.3130 | +4.6% / +2.7% | discard | neutral vs same-load control (1.008); lane-16 lowering already near-optimal on M1 NEON; reverted |
