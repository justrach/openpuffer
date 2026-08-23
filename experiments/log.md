# experiments/log.md — zvec optimization ledger

Metric: `zig build -Doptimize=ReleaseFast && ./zig-out/bin/zvec bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128`
Score: lowest `zvec (ANN) p50 (ms)` subject to `zvec recall@10 >= 0.30`. Gate: `zig build test` must pass.
Protocol: see `program.md`. Append-only; one row per experiment; E000 baseline seeded by human, never edited.

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | notes |
|----|------|------------|----------------|---------------|------|----------|-----------|-------------------|---------|-------|
| E000 | 2026-08-23 | baseline (seeded by human, never edit) | none — unmodified tree | — | pass | 1.013 / 0.981 | 0.3130 | — | keep | two consecutive runs; ±3% p50 variance, recall bit-stable; M1 Mac |
