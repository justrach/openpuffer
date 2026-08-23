# zvec autonomous optimization loop

This repo runs an autonomous research loop in the style of
[karpathy/autoresearch](https://github.com/karpathy/autoresearch): an agent
repeatedly proposes a single change to one editable file, measures it against a
fixed metric and gate, keeps it if the metric improved, reverts it if not, and
logs every outcome to an append-only ledger. Autoresearch optimized `val_bpb`
in `train.py` under a 5-minute budget; here we optimize p50 ANN query latency
in `src/hnsw.zig` (+ `src/vector.zig`) under a fixed bench command.

## Roles of files

- **`src/hnsw.zig`** — THE file you edit (analogous to `train.py`). HNSW index:
  graph construction, search, int8 traversal + f32 rerank. `src/vector.zig`
  (SIMD dot product / cosine distance) may also be edited when your hypothesis
  lives at that layer — treat the pair as one unit.
- **Fixed infrastructure — DO NOT MODIFY** (analogous to `prepare.py`):
  `src/main.zig` (CLI + benchmark harness), `build.zig`, `README.md`,
  `src/gemini.zig`, `src/tpuf.zig`. Another agent/peer may be editing these;
  touching them invalidates the metric and collides with their work.
- **`experiments/log.md`** — the ledger. Append one row per experiment. Never
  rewrite or delete rows.

## THE METRIC

Build and run exactly this, no parameter changes:

```
zig build -Doptimize=ReleaseFast
./zig-out/bin/zvec bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128 > /tmp/zvec-bench.log 2>&1
grep -E "p50|recall" /tmp/zvec-bench.log
```

Output lines look like (parse these mechanically):

```
zvec (ANN)     p50   1.029ms  p95   1.200ms  p99   1.275ms  mean   1.042ms  191979.1 qps
zvec recall@10 vs exact: 0.3130
exact scan     p50   3.402ms  ...
```

Score rules:
- **Primary score: the `zvec (ANN) p50` value in ms. Lower is better.**
- **Hard floor: `zvec recall@10` must be ≥ 0.30** on this random dataset
  (measured baseline is 0.3130; uniformly random high-dim vectors are ANN worst
  case per README). A faster result below the floor is a discard.
- Ignore the `exact scan` line except as context.
- **Noise floor: measured run-to-run p50 variance is ±3% on this machine.** A
  single-run delta under 4% is NOT an improvement — re-run once; only keep if
  the improvement reproduces across both runs. Recall@10 is bit-stable, so any
  recall change is real (and any drop below the floor is a discard).
- Compare on the same machine state as the previous best; note anomalies
  (thermal load, background builds) in the ledger notes column.

Reference baseline (M1 Mac): p50 ≈ 1.03–1.08 ms, recall@10 = 0.3130.

## VERSION CONTROL (step 0 — do this before your first cycle)

This directory may or may not be a git repository. Check for `.git`:

- **If git exists**: use it. Commit a baseline (`git add -A && git commit -m
  "baseline"`) if the tree is dirty, then follow the git flow in step 7.
- **If not** (current state as of seeding): initialize it — `git init && git
  add -A && git commit -m "baseline"`. This is safe and non-destructive; the
  keep/revert discipline below depends on it existing.
- If you cannot create a repo at all, fall back to snapshots: before each
  edit, `cp src/hnsw.zig src/vector.zig /tmp/zvec-snap/` and revert by copying
  back. Say which mode you're in in each ledger row's notes.

## THE GATE

Before any run counts, `zig build test` must pass. If tests fail, fix within
the current experiment (a test failure caused by your edit is not a crash to
log separately unless you give up on the hypothesis).

## One experiment cycle

Loop forever until interrupted. Each cycle:

1. **Hypothesis**: pick ONE code-change hypothesis (e.g. "reserve neighbor
   buffer capacity to skip allocator churn", "widen SIMD in vector.zig").
2. **Log first** (see Budget rule below): write the row skeleton into
   `experiments/log.md` before editing.
3. **Edit** `src/hnsw.zig` and/or `src/vector.zig` only.
4. **Gate**: `zig build test`. Fail → fix or revert, log verdict.
5. **Bench**: build ReleaseFast, run the metric command, parse p50 + recall@10.
6. **Compare vs best-so-far**: keep only if recall@10 ≥ 0.30 AND p50 < best p50.
7. **Keep or revert**: if improved (recall floor met AND p50 win reproduced
   past the noise floor), `git add src/hnsw.zig src/vector.zig
   experiments/log.md && git commit` — this advances the lineage. If equal or
   worse, revert the edit (`git checkout -- src/hnsw.zig src/vector.zig`, or
   restore from `/tmp/zvec-snap/` in snapshot mode), update the log row with
   verdict `discard` and why, and do NOT commit the change (commit only the
   log row).
8. Go to 1. Never stop to ask whether to continue — you are autonomous.

## Budget rule

Each bench run is seconds, not minutes, so the autoresearch 5-minute budget
maps to: **ONE hypothesis per experiment cycle, and at most N=2 bench runs per
hypothesis** (one measurement + one retry after a trivial fix). If two runs
don't produce a clean number, log `discard`/`crash` with notes and move on.
A new idea requires the previous idea's ledger entry to be complete first.

## Ledger format (`experiments/log.md`)

One markdown table row per experiment:

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | notes |

- `id`: zero-padded sequence `E001`, `E002`, …
- `gate`: `pass` / `fail`.
- `delta p50 vs best`: signed % vs best-so-far entering the experiment.
- `verdict`: `keep` (new best, committed) / `discard` (reverted) / `crash`.
- Baseline row `E000` is seeded by the human; never edit it.

## Guardrails (adapted from autoresearch)

- **Never game the metric**: do not reduce `--queries`, `--n`, `--dim`,
  `--k`, or `--ef`; do not alter the bench command; do not modify the harness in
  `src/main.zig` to flatter results. Disabling/bypassing rerank for raw speed
  is allowed ONLY if the resulting recall@10 is logged honestly and still
  clears the 0.30 floor.
- **Don't touch peer-owned files**: anything outside `src/hnsw.zig`,
  `src/vector.zig`, `experiments/log.md` is off-limits.
- **Commit discipline**: commit source changes only after ≥1 kept improvement
  establishes a lineage point; discards are reverted, not committed (except the
  ledger row itself).
- **Simplicity criterion**: all else equal, simpler is better. A 5% p50 win via
  30 lines of hacky special-casing? Probably not worth it. Equal performance
  with less code? Keep.
- Crashes/build breaks: fix trivial ones (typo, missing import); if
  fundamentally broken after the retry budget, log `crash`, revert, move on.
