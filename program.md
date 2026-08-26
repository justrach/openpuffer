# openpuffer autonomous optimization loop

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
./zig-out/bin/openpuffer bench-synthetic --n 20000 --queries 200 --dim 1536 --k 10 --ef 128 > /tmp/openpuffer-bench.log 2>&1
grep -E "p50|recall" /tmp/openpuffer-bench.log
```

Output lines look like (parse these mechanically):

```
openpuffer (ANN)     p50   0.981ms  p95   1.138ms  p99   1.333ms  mean   0.995ms  200915.2 qps
openpuffer recall@10 vs exact: 0.3130
exact scan     p50   3.355ms  p95   3.498ms  p99   3.590ms  mean   3.345ms   59790.3 qps
```

Score rules:
- **Primary score: the `openpuffer (ANN) p50` value in ms. Lower is better.**
- **Hard floor: `openpuffer recall@10` must be ≥ 0.30** on this random dataset
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
- **Re-baseline rule (added after E007)**: run-to-run ambient drift on this
  machine can reach ±5%, larger than most candidate effects. Before trusting
  any sub-8% comparison, re-measure the CURRENT BEST tree (revert your edit,
  bench twice) and compare against THAT number, not the historical best. Log
  both control values in the row's notes. Historical bests go stale.

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
  git init && git add -A && git -c user.name=openpuffer-agent -c user.email=agent@local commit -m "baseline"
  ```

  The inline `-c` flags set an identity for this one command WITHOUT touching
  global config — required because fresh machines often have none and `git
  commit` would otherwise fail. This is safe and non-destructive.
- **Snapshot fallback** (only if you cannot create a repo at all): before each
  edit run `mkdir -p /tmp/openpuffer-snap/ && cp src/hnsw.zig /tmp/openpuffer-snap/<ID>-hnsw.zig`
  (same for vector.zig), using the experiment's id so each experiment has its
  own restore point; revert by copying back. Say which mode you're in in each
  ledger row's notes.

## THE GATE

Before any run counts, `zig build test` must pass. That step now runs the
HNSW tests in `src/hnsw.zig` (imported-file tests are otherwise skipped).
If tests fail, fix within the current experiment (a test failure caused by
your edit is not a crash to log separately unless you give up on the
hypothesis).

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
   experiments/log.md && git -c user.name=openpuffer-agent -c user.email=agent@local commit -m "E00N: <one-line result>"`
   — this advances the lineage. If equal or worse, FIRST revert the source
   edit (`git checkout -- src/hnsw.zig src/vector.zig`, or restore from your
   `/tmp/openpuffer-snap/<ID>-*` copies in snapshot mode), THEN update the log row
   with verdict `discard` and why, and commit ONLY the ledger file — the
   discard commit must never contain reverted source changes.
8. Go to 1. Never stop to ask whether to continue — you are autonomous.
9. **Regenerate the chart** after closing a row: `python3 tools/plot_results.py`
   rewrites `experiments/results.svg` from the ledger; include it in the same
   commit as the ledger row.

## Budget rule

Each bench run is seconds, not minutes, so the autoresearch 5-minute budget
maps to: **ONE hypothesis per experiment cycle, and at most N=2 bench runs per
hypothesis** (Run 1 + Run 2 of the noise-floor decision procedure; a trivial
fix may land between them). If both runs fail to produce a clean number, log
`discard`/`crash` with notes and move on.
A new idea requires the previous idea's ledger entry to be complete first.

## Ledger format (`experiments/log.md`)

One markdown table row per experiment:

| id | date | hypothesis | change summary | files touched | gate | p50 (ms) | recall@10 | delta p50 vs best | verdict | track | notes |

- `id`: zero-padded sequence `E001`, `E002`, …
- `gate`: `pass` / `fail`.
- `delta p50 vs best`: signed % vs best-so-far **on the same track**.
- `verdict`: `running` (row opened, no result yet) / `keep` (new best,
  committed) / `discard` (reverted) / `crash`.
- `track`: `engine` | `serve` | `memory` | `knobs` | `scale`. Only
  `engine` numeric p50s go into `results.svg` and the 4% keep bar.
- Baseline row `E000` is seeded by the human; never edit it.

## Guardrails (adapted from autoresearch)

- **Never game the metric**: do not reduce `--queries`, `--n`, `--dim`,
  `--k`, or `--ef`; do not alter the bench command; do not modify the harness in
  `src/main.zig` to flatter results. Disabling/bypassing rerank for raw speed
  is allowed ONLY if the resulting recall@10 is logged honestly and still
  clears the 0.30 floor.
- **Don't touch peer-owned files**: stay inside the files listed for your
  **track** (see 2026-08-25 amendment). `experiments/log.md` is always
  writable. Do not change the scored bench command.
- **Commit discipline**: EVERY kept improvement is committed, including the
  very first one — the first keep IS the lineage point. Discards are reverted,
  not committed (except the ledger row itself).
- **Simplicity criterion**: all else equal, simpler is better. A 5% p50 win via
  30 lines of hacky special-casing? Probably not worth it. Equal performance
  with less code? Keep.
- Crashes/build breaks: fix trivial ones (typo, missing import); if
  fundamentally broken after the retry budget, log `crash`, revert, move on.

## Amendments

### 2026-08-23 — real-data metric + serving-path scope (human-authorized)

1. **Secondary real-data metric registered**: `python3 tools/qa_bench.py`
   benchmarks SQuAD dev passages/questions through `openpuffer serve` vs
   turbopuffer cloud (latency, ANN recall vs exact, gold-passage hit@k).
   It does NOT replace the scored `bench-synthetic` p50; use it to confirm
   that engine changes don't regress real-world retrieval quality
   (measured 2026-08-23: recall@10 0.9997, gold-hit@10 1.0000 at 2k docs).
2. **Scope expansion**: `src/server.zig`'s query path (`handleQuery`,
   `Options.ef`) may now be edited for serving-latency experiments,
   CONDITIONAL on the peer session having no uncommitted changes to that
   file (`git status --short src/server.zig` must be clean before editing).
   The same gate/ledger/noise-floor rules apply; ledger rows note that p50
   includes HTTP+JSON overhead.
3. Measured context for future experiments: local serve p50 is ~1.31ms vs
   ~0.83ms direct library call at ef=128 — roughly 0.5ms of HTTP+JSON
   per-request overhead is the largest remaining target, bigger than any
   remaining engine-side effect (E010/E011 confirmed the traversal loop is
   at a plateau: prefetch-fed dot products dominate, everything else is
   below this machine's noise floor).

### 2026-08-25 — tracks, real gate, loop driver (human-authorized)

The original one-file / one-metric loop plateaued and then drifted: HTTP
p50s, flatten RSS, and knob rows were logged as if they were 20k engine
p50. `zig build test` compiled `src/main.zig` (no tests) and never ran
`src/hnsw.zig`. Agents also edited `src/main.zig` / `src/server.zig`
despite the freeze.

1. **Tracks.** Every ledger row has a `track` cell. `tools/plot_results.py`
   plots **engine only**. A `keep` on `serve` / `memory` / `knobs` / `scale`
   does **not** move the engine best or the 4% bar. Put HTTP / 1M / RSS
   numbers in notes or the Scale/Pareto sections, never in an engine p50
   cell.
2. **Gate.** `zig build test` now runs `src/hnsw.zig` tests. Confirm with
   `zig test src/hnsw.zig -OReleaseFast` if you want the ReleaseFast suite.
3. **Driver.** After an edit, run `python3 tools/loop_once.py`. It executes
   the exact scored command, parses p50 + recall@10, and prints keep/discard
   vs the best **engine** keep. It does not pick a hypothesis.
4. **Editable files by track** (still one hypothesis per cycle):
   - `engine`: `src/hnsw.zig`, `src/vector.zig`
   - `serve`: `src/server.zig`, `src/iouring_sock.zig` (tree must be clean)
   - `memory` / `scale`: `src/hnsw.zig` plus harness flags already on main
   - `knobs`: `src/main.zig` / `src/server.zig` only to *expose* an existing
     engine knob — not to change the scored command
5. **Do not re-run E002–E022.** Those inner-loop ideas are spent. Engine
   work now needs a new mechanism (e.g. AMX/VNNI `dotI8`), not another
   prefetch width.
6. **Backlog (pick one, in this order, unless a human redirects):**
   1. `memory`: drop the hot-path f32 copy (int8-only / optional rerank) so
      2M can fit on 16 GiB after flatten (1M is 7.3 GiB).
   2. `memory`: mmap the flat slabs.
   3. `serve`: RCU / finer lock — mixed writers still stall readers after
      fair-rotate (exclusive splice).
   4. `serve`: Zig/io_uring shard router with keep-alive to children
      (Python hop is ~0.46 ms; same-box N shards lose QPS).
   5. `scale`: clustered 1M recall@10 at ef=64/128 (`--clustered --no-exact`
      is latency-only today).
   6. `engine`: AMX or AVX-512 VNNI `dotI8` only if it can clear 4% vs
      **E022 0.575 ms** on this host after a fresh control pair.

Clustered (not uniform random) is the product quality axis. Serve default
ef=128 stays. Random 1536-d cannot hold recall@10 ≥ 0.30 at n=200k even at
ef=1024 — do not spend engine budget chasing that cliff.

### 2026-08-26 — hardware keys, macOS RSS, codedb profile

1. **Baseline keys.** `tools/loop_once.py` compares only against engine
   keeps with a compatible `{os}-{arch}-{cpu}/{profile}` key
   (`tools/bench_key.py`). E000–E011 infer `darwin-arm64-apple-m1`;
   E012+ engine infers `linux-x86_64-intel-xeon`. A new host (M4, …)
   prints `baseline (no compatible keep)` instead of judging against
   Xeon. Verdicts include `key=`. Small wins still want a fresh control
   (`OPENPUFFER_CONTROL_P50`).
2. **RSS.** `src/rss.zig` reports **current** RSS on Linux (`VmRSS`) and
   macOS (`resident_size`). Peak is `VmHWM` / `resident_size_max` when
   present. `bench-synthetic` prints `rss_mib phase=… current=…` for
   gates. `unavailable` only if the probe fails.
3. **codedb profile.** `python3 tools/codedb_once.py` is a separate
   512-d / k=24 / clustered ledger (`experiments/codedb.md`). It does
   not move the 1536-d engine best. Recall@24 floor is 0.99. This is an
   ANN engine number; embedder/GPU time is measured elsewhere.
4. **codedb real-repository gate.** `python3 tools/codedb_repo_bench.py`
   builds or consumes external, secret-filtered Qwen3 0.6B/512D caches and derives
   agent-like queries from codedb commit subjects. Exact cosine top-24 is the
   ANN ground truth (floor 0.99); changed-file hit/MRR/nDCG are reported as
   separate embedding-quality diagnostics. Configuration sweeps are
   interleaved to control machine drift. The measured codedb caller profile is
   `ef=48, rerank_mult=2`; it does not change the generic server default.

GHA `macos-latest` / `ubuntu-latest` (2026-08-26, PR #22, main tree
without later engine keeps):

| host key | engine p50 | recall@10 | post-query RSS | codedb p50 | recall@24 | codedb RSS |
|---|---|---|---|---|---|---|
| darwin-arm64-apple-m1 | 1.030 ms | 0.3105 | 310 MiB current/peak | 0.173 ms | 1.0000 | 185 MiB |
| linux-x86_64 (GHA) | 0.852 ms | 0.3105 | 363 MiB current/peak | 0.123 ms | 1.0000 | 219 MiB |

macOS printed numeric `rss_mib` (`resident_size`), not `unavailable`.
GHA M1 is compared to E009 0.830, not Xeon E022 0.575. Do not mix
these with the Linux Xeon flatten+RCU 0.433 ms keep.
