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
zvec (ANN)     p50   0.981ms  p95   1.138ms  p99   1.333ms  mean   0.995ms  200915.2 qps
zvec recall@10 vs exact: 0.3130
exact scan     p50   3.355ms  p95   3.498ms  p99   3.590ms  mean   3.345ms   59790.3 qps
```

Score rules:
- **Primary score: the `zvec (ANN) p50` value in ms. Lower is better.**
- **Hard floor: `zvec recall@10` must be ≥ 0.30** on this random dataset
  (measured baseline is 0.3130; uniformly random high-dim vectors are ANN worst
  case per README). A faster result below the floor is a discard.
- Ignore the `exact scan` line except as context.
- **Noise floor: measured run-to-run p50 variance is ±3% on this machine, so
  the decision procedure per hypothesis is exactly:**
  1. Run 1 measures. Delta ≥ 4% faster than best AND recall floor met → keep,
     no further runs needed.
  2. Otherwise Run 2 decides: ≥ 4% faster than best → keep, else discard.
  3. A trivial-fix rebuild between runs is allowed but consumes the budget —
     there is no third run. If the fix isn't obviously mechanical, discard.

  Recall@10 is bit-stable, so ANY recall change is real (and a drop below the
  floor is an instant discard regardless of speed).
- Compare on the same machine state as the previous best; note anomalies
  (thermal load, background builds) in the ledger notes column.

Reference baseline (M1 Mac, 2026-08-23): p50 0.981–1.013 ms across runs,
recall@10 = 0.3130. Best-so-far entering E001 is **0.981 ms** (the lower of
E000's two runs — see experiments/log.md).

## VERSION CONTROL (step 0 — do this before your first cycle)

Check for `.git` in the repo root:

- **This repo HAS `.git`** with a baseline commit — verify with `git log
  --oneline | head -1`. If the tree is dirty from someone else's work-in-
  progress, leave their files alone; just note it in your first row's notes.
- **If `.git` is missing**, create it:

  ```
  git init && git add -A && git -c user.name=zvec-agent -c user.email=agent@local commit -m "baseline"
  ```

  The inline `-c` flags set an identity for this one command WITHOUT touching
  global config — required because fresh machines often have none and `git
  commit` would otherwise fail. This is safe and non-destructive.
- **Snapshot fallback** (only if you cannot create a repo at all): before each
  edit run `mkdir -p /tmp/zvec-snap/ && cp src/hnsw.zig /tmp/zvec-snap/<ID>-hnsw.zig`
  (same for vector.zig), using the experiment's id so each experiment has its
  own restore point; revert by copying back. Say which mode you're in in each
  ledger row's notes.

## THE GATE

Before any run counts, `zig build test` must pass. If tests fail, fix within
the current experiment (a test failure caused by your edit is not a crash to
log separately unless you give up on the hypothesis).

## One experiment cycle

Loop forever until interrupted. Each cycle:

1. **Hypothesis**: pick ONE code-change hypothesis (e.g. "reserve neighbor
   buffer capacity to skip allocator churn", "widen SIMD in vector.zig").
2. **Log first** (see Budget rule below): append a PENDING row skeleton to
    `experiments/log.md` BEFORE editing — e.g.
    `| E002 | <date> | <hypothesis> | <change summary> | <files> | pending | — | — | — | running | |`
    — then fill the cells in as results arrive. Only the CURRENT experiment's
    row may be rewritten; once its verdict is final, the row is immutable.
3. **Edit** `src/hnsw.zig` and/or `src/vector.zig` only.
4. **Gate**: `zig build test`. Fail → fix or revert, log verdict.
5. **Bench**: build ReleaseFast, run the metric command, parse p50 + recall@10.
6. **Compare vs best-so-far**: keep only if recall@10 ≥ 0.30 AND p50 < best p50.
7. **Keep or revert**: if improved (recall floor met AND a ≥4% p50 win per the
   decision procedure above), `git add src/hnsw.zig src/vector.zig
   experiments/log.md && git -c user.name=zvec-agent -c user.email=agent@local commit -m "E00N: <one-line result>"`
   — this advances the lineage. If equal or worse, FIRST revert the source
   edit (`git checkout -- src/hnsw.zig src/vector.zig`, or restore from your
   `/tmp/zvec-snap/<ID>-*` copies in snapshot mode), THEN update the log row
   with verdict `discard` and why, and commit ONLY the ledger file — the
   discard commit must never contain reverted source changes.
8. Go to 1. Never stop to ask whether to continue — you are autonomous.

## Budget rule

Each bench run is seconds, not minutes, so the autoresearch 5-minute budget
maps to: **ONE hypothesis per experiment cycle, and at most N=2 bench runs per
hypothesis** (Run 1 + Run 2 of the noise-floor decision procedure; a trivial
fix may land between them). If both runs fail to produce a clean number, log
`discard`/`crash` with notes and move on.
A new idea requires the previous idea's ledger entry to be complete first.

## Ledger format (`experiments/log.md`)

One markdown table row per experiment:

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | notes |

- `id`: zero-padded sequence `E001`, `E002`, …
- `gate`: `pass` / `fail`.
- `delta p50 vs best`: signed % vs best-so-far entering the experiment.
- `verdict`: `running` (row opened, no result yet) / `keep` (new best,
  committed) / `discard` (reverted) / `crash`.
- Baseline row `E000` is seeded by the human; never edit it.

## Guardrails (adapted from autoresearch)

- **Never game the metric**: do not reduce `--queries`, `--n`, `--dim`,
  `--k`, or `--ef`; do not alter the bench command; do not modify the harness in
  `src/main.zig` to flatter results. Disabling/bypassing rerank for raw speed
  is allowed ONLY if the resulting recall@10 is logged honestly and still
  clears the 0.30 floor.
- **Don't touch peer-owned files**: anything outside `src/hnsw.zig`,
  `src/vector.zig`, `experiments/log.md` is off-limits.
- **Commit discipline**: EVERY kept improvement is committed, including the
  very first one — the first keep IS the lineage point. Discards are reverted,
  not committed (except the ledger row itself).
- **Simplicity criterion**: all else equal, simpler is better. A 5% p50 win via
  30 lines of hacky special-casing? Probably not worth it. Equal performance
  with less code? Keep.
- Crashes/build breaks: fix trivial ones (typo, missing import); if
  fundamentally broken after the retry budget, log `crash`, revert, move on.
