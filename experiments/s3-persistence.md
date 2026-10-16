# S3 persistence under load and faults

Restarting from object storage could silently lose namespaces. This records
what a multi-namespace load test against an S3-compatible store found, what
changed, and the gate that now guards it.

## Gate

```sh
zig build -Doptimize=ReleaseFast
python3 tools/s3_persistence_gate.py --bin zig-out/bin/openpuffer
```

Stdlib-only Python. It starts `tools/s3_fake.py` (a path-style S3 stand-in
with real `ListObjectsV2` paging, keep-alive and switchable faults) and the
server on free ports, then restarts the server under faults. After every
restart each namespace must be either exact (same document count, same top-10
for fixed probes) or refused with 503 while quarantined. It must never be
served partially, and nothing already persisted may be lost.

| Scenario | Faults | Must hold |
|---|---|---|
| pagination | bucket spans several LIST pages | every namespace recovers |
| write-faults | 20% of PUT/DELETE answer 503, 5% of connections drop | no acknowledged write lost; S3 connections reused |
| snapshot-fails | every compaction snapshot PUT fails | WAL kept; exact after restart; compaction succeeds once the store recovers |
| quarantine | one namespace's GETs fail at restart | 503 for it only; heals in the background without a restart |
| read-faults | 60% of GETs fail at restart | none served partially; all heal; a clean restart still matches |
| list-fails | LIST fails | server refuses to start; bucket untouched |

It runs in CI on Linux (io_uring server) and macOS.

## Findings and fixes

1. **One LIST page.** `listKeys` read the first 1000 keys only, so recovery
   skipped every namespace past them. It now follows `NextContinuationToken`.
2. **A transient GET read as "not found".** `getObjectOrNull` mapped any
   failed GET to null. A 503 on `snapshot.bin` made recovery rebuild from the
   WAL tail and then compact that partial index over the snapshot. Only a 404
   is absence now.
3. **Half-loaded namespaces stayed live.** After a failed recovery the next
   write reused an existing WAL sequence number, or compaction overwrote the
   snapshot. Failed namespaces are quarantined (503 for reads and writes) and
   reloaded in the background into a fresh namespace that is swapped in only
   when complete.
4. **WAL deleted after a failed snapshot.** Compaction ignored a failed
   snapshot PUT and deleted the WAL anyway, losing every document since the
   previous snapshot. The WAL is now deleted only after the snapshot lands;
   otherwise compaction is re-armed for the next flush.
5. **Unlistable bucket.** Startup served empty and accepted writes at WAL
   sequence 1. It now exits.
6. **Connections.** Every S3 request opened a new TCP connection, recovery
   started one thread and connection per namespace, and all WAL uploads shared
   one lock. Requests now keep connections alive and retry transport errors,
   429 and 5xx (5 attempts, 50 ms doubling backoff; every call is
   idempotent). Recovery uses a bounded pool, and WAL uploads use lanes; a
   namespace always maps to the same lane, so its segments stay ordered.

## Measurements

300 namespaces × 200 vectors (768-d embeddings), 25 ms per fake-S3 request,
16 writers then 32 mixed clients; macOS arm64, ReleaseFast.

| | before | after |
|---|---|---|
| namespaces back after restart | 198 / 300 | 300 / 300 (counts and 900 probe top-10s exact) |
| S3 connections, write + soak | 1921 | 12 |
| write p50 / p99 | 463 / 735 ms | 176 / 477 ms |
| queries/s with 5% writes mixed in | 429 | 1273 |
| restart (recovery) time | 1.0 s | 2.6 s |

Restart while 30% of GETs fail, then 8 writes per namespace, then a clean
restart (20 namespaces): before, 15 lost documents; after, 0.

Recovery is slower than unbounded fan-out because it is capped (default 32
workers) and compacts as it loads; the old speed came from a connection per
namespace and lost a third of them.

## Knobs

| Variable | Default | Meaning |
|---|---|---|
| `OPENPUFFER_RECOVERY_WORKERS` | 32 | parallel namespace loads at startup |
| `OPENPUFFER_PERSIST_LANES` | 8 | WAL upload lanes, one S3 connection each |
| `OPENPUFFER_HEAL_INITIAL_MS` | 1000 | first background retry for a quarantined namespace (doubles to 30 s) |

## Not covered here

On Linux, an idle keep-alive client connection holds an io_uring worker in a
blocking `recv`. With as many idle connections as workers, other clients get
no answer. The gate closes idle connections so it is not affected; the server
fix is separate.
