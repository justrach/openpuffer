#!/usr/bin/env python3
"""Mixed live R/W workload (no API keys).

Starts `openpuffer serve`, loads a modest base set, then for a fixed window
runs HTTP keep-alive threads in parallel on the same namespace:

  - query workers  (ANN)
  - insert workers (new ids, same upsert API)
  - update workers (existing ids, same upsert API)

Compares that mixed window against a query-only control on the same corpus.

Serve holds `std.Io.RwLock` exclusive around HNSW insert/update in
`handleWrite`, and shared during `handleQuery`. This bench is the first
measurement of query p50 under concurrent writers.

Keep the corpus modest (n=2000–20000, dim=1536). Do not use this to storm
1–2M HTTP upserts.

Usage:
    python3 tools/mixed_bench.py
    python3 tools/mixed_bench.py --n 8000 --dim 1536 --seconds 8 --port 8701
    python3 tools/mixed_bench.py --n 2000 --dim 256 --seconds 6 --port 8702
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request


def http_json(url, payload=None, method=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def unit_vec(dim, rng):
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def percentiles(samples):
    if not samples:
        return None, None, None
    s = sorted(samples)

    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))]

    return pct(0.50), pct(0.95), pct(0.99)


def query_payload(qv, k, ef=None):
    p = {"rank_by": ["vector", "ANN", qv], "top_k": k}
    if ef is not None:
        p["ef"] = ef
    return p


def upsert_payload(ids, vecs):
    return {
        "upsert_columns": {"id": ids, "vector": vecs},
        "distance_metric": "cosine_distance",
    }


def ms(x):
    return None if x is None else x * 1000.0


class Keepalive:
    def __init__(self, host, port, timeout):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.conn = http.client.HTTPConnection(host, port, timeout=timeout)

    def post(self, path, payload):
        body = json.dumps(payload).encode()
        t0 = time.perf_counter()
        try:
            self.conn.request(
                "POST", path, body=body, headers={"content-type": "application/json"}
            )
            resp = self.conn.getresponse()
            data = resp.read()
            elapsed = time.perf_counter() - t0
            if resp.status != 200:
                return elapsed, False, f"http_{resp.status}"
            return elapsed, True, None
        except socket.timeout:
            self._reconnect()
            return time.perf_counter() - t0, False, "timeout"
        except Exception as e:
            self._reconnect()
            return time.perf_counter() - t0, False, type(e).__name__

    def _reconnect(self):
        try:
            self.conn.close()
        except Exception:
            pass
        self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


class Stats:
    def __init__(self):
        self._mu = threading.Lock()
        self.query_ok = []
        self.query_err = []
        self.insert_ok = []
        self.insert_err = []
        self.update_ok = []
        self.update_err = []
        self.err_kinds = {}

    def add(self, kind, elapsed, ok, err):
        with self._mu:
            bucket = {
                ("query", True): self.query_ok,
                ("query", False): self.query_err,
                ("insert", True): self.insert_ok,
                ("insert", False): self.insert_err,
                ("update", True): self.update_ok,
                ("update", False): self.update_err,
            }[(kind, ok)]
            bucket.append(elapsed)
            if err:
                self.err_kinds[err] = self.err_kinds.get(err, 0) + 1

    def snapshot(self):
        with self._mu:
            return {
                "query_ok": list(self.query_ok),
                "query_err": list(self.query_err),
                "insert_ok": list(self.insert_ok),
                "insert_err": list(self.insert_err),
                "update_ok": list(self.update_ok),
                "update_err": list(self.update_err),
                "err_kinds": dict(self.err_kinds),
            }


class IdSpace:
    """Existing ids are 0..high-1. Inserts atomically take the next id."""

    def __init__(self, n):
        self._mu = threading.Lock()
        self.high = n

    def next_insert_ids(self, count):
        with self._mu:
            start = self.high
            self.high += count
            return list(range(start, start + count))

    def random_existing(self, rng, count):
        with self._mu:
            high = self.high
        if high <= 0:
            return []
        return [rng.randrange(high) for _ in range(count)]


def wait_ready(port, server, timeout=8.0):
    deadline = time.perf_counter() + timeout
    url = f"http://127.0.0.1:{port}/v2/namespaces/"
    while time.perf_counter() < deadline:
        try:
            urllib.request.urlopen(url, timeout=0.2)
            return
        except urllib.error.HTTPError:
            return
        except Exception:
            if server.poll() is not None:
                out = server.stdout.read().decode() if server.stdout else ""
                sys.exit(f"server died:\n{out}")
            time.sleep(0.05)
    sys.exit("server did not become ready")


def load_docs(base, docs, batch):
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
        http_json(base, upsert_payload(list(range(i, i + len(chunk))), chunk), timeout=120)
    return time.perf_counter() - t0


def query_worker(stop, host, port, ns, queries, k, ef, timeout, stats):
    client = Keepalive(host, port, timeout)
    path = f"/v2/namespaces/{ns}/query"
    rng = random.Random(threading.get_ident() ^ int(time.time() * 1e6))
    try:
        while not stop.is_set():
            qv = queries[rng.randrange(len(queries))]
            elapsed, ok, err = client.post(path, query_payload(qv, k, ef))
            stats.add("query", elapsed, ok, err)
    finally:
        client.close()


def write_worker(kind, stop, host, port, ns, dim, batch, timeout, stats, ids, vec_rng_seed):
    client = Keepalive(host, port, timeout)
    path = f"/v2/namespaces/{ns}"
    rng = random.Random(vec_rng_seed)
    try:
        while not stop.is_set():
            vecs = [unit_vec(dim, rng) for _ in range(batch)]
            if kind == "insert":
                id_list = ids.next_insert_ids(batch)
            else:
                id_list = ids.random_existing(rng, batch)
                if not id_list:
                    time.sleep(0.001)
                    continue
            elapsed, ok, err = client.post(path, upsert_payload(id_list, vecs))
            stats.add(kind, elapsed, ok, err)
    finally:
        client.close()


def run_window(label, seconds, host, port, ns, queries, args, ids, insert_n, update_n):
    stats = Stats()
    stop = threading.Event()
    threads = []
    for i in range(args.query_workers):
        threads.append(
            threading.Thread(
                target=query_worker,
                args=(stop, host, port, ns, queries, args.k, args.ef, args.timeout, stats),
                name=f"q{i}",
                daemon=True,
            )
        )
    for i in range(insert_n):
        threads.append(
            threading.Thread(
                target=write_worker,
                args=("insert", stop, host, port, ns, args.dim, args.write_batch, args.timeout, stats, ids, 1000 + i),
                name=f"ins{i}",
                daemon=True,
            )
        )
    for i in range(update_n):
        threads.append(
            threading.Thread(
                target=write_worker,
                args=("update", stop, host, port, ns, args.dim, args.write_batch, args.timeout, stats, ids, 2000 + i),
                name=f"upd{i}",
                daemon=True,
            )
        )
    wall0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=args.timeout + 2)
    wall = time.perf_counter() - wall0
    snap = stats.snapshot()
    return summarize(label, wall, snap, ids.high)


def summarize(label, wall, snap, corpus_n):
    q_ok = snap["query_ok"]
    q_err = snap["query_err"]
    i_ok = snap["insert_ok"]
    i_err = snap["insert_err"]
    u_ok = snap["update_ok"]
    u_err = snap["update_err"]
    q_all = q_ok + q_err
    w_ok = i_ok + u_ok
    w_err = i_err + u_err
    qp50, qp95, qp99 = percentiles(q_ok)
    wp50, wp95, wp99 = percentiles(w_ok)
    ip50, _, _ = percentiles(i_ok)
    up50, _, _ = percentiles(u_ok)
    q_n = len(q_all)
    w_n = len(w_ok) + len(w_err)
    q_err_rate = (len(q_err) / q_n) if q_n else 0.0
    w_err_rate = (len(w_err) / w_n) if w_n else 0.0
    qps = len(q_ok) / wall if wall > 0 else 0.0
    wps = len(w_ok) / wall if wall > 0 else 0.0
    qmax = max(q_all) if q_all else None
    stall = None
    if q_ok and qp50:
        stall = sum(1 for x in q_ok if x >= 2.0 * qp50) / len(q_ok)
    return {
        "label": label,
        "wall_s": wall,
        "corpus_n": corpus_n,
        "query": {
            "n": q_n,
            "ok": len(q_ok),
            "errors": len(q_err),
            "error_rate": q_err_rate,
            "p50_ms": ms(qp50),
            "p95_ms": ms(qp95),
            "p99_ms": ms(qp99),
            "mean_ms": ms(statistics.mean(q_ok) if q_ok else None),
            "max_ms": ms(qmax),
            "qps": qps,
            "frac_ge_2x_p50": stall,
        },
        "write": {
            "n": w_n,
            "ok": len(w_ok),
            "errors": len(w_err),
            "error_rate": w_err_rate,
            "p50_ms": ms(wp50),
            "p95_ms": ms(wp95),
            "p99_ms": ms(wp99),
            "wps": wps,
            "insert_ok": len(i_ok),
            "insert_p50_ms": ms(ip50),
            "update_ok": len(u_ok),
            "update_p50_ms": ms(up50),
        },
        "err_kinds": snap["err_kinds"],
    }


def print_phase(r):
    q = r["query"]
    w = r["write"]
    print(f"--- {r['label']}  wall={r['wall_s']:.2f}s  corpus_n={r['corpus_n']} ---")
    print(
        f"query  n={q['n']:5d} ok={q['ok']:5d} err={q['errors']} ({q['error_rate']*100:.2f}%)  "
        f"p50 {fmt_ms(q['p50_ms'])}  p95 {fmt_ms(q['p95_ms'])}  p99 {fmt_ms(q['p99_ms'])}  "
        f"max {fmt_ms(q['max_ms'])}  qps={q['qps']:.1f}"
    )
    if w["n"]:
        print(
            f"write  n={w['n']:5d} ok={w['ok']:5d} err={w['errors']} ({w['error_rate']*100:.2f}%)  "
            f"p50 {fmt_ms(w['p50_ms'])}  p95 {fmt_ms(w['p95_ms'])}  "
            f"insert_p50 {fmt_ms(w['insert_p50_ms'])} ({w['insert_ok']})  "
            f"update_p50 {fmt_ms(w['update_p50_ms'])} ({w['update_ok']})  "
            f"wps={w['wps']:.1f}"
        )
    else:
        print("write  (none)")
    if r["err_kinds"]:
        print(f"errors {r['err_kinds']}")


def fmt_ms(x):
    return "   n/a  " if x is None else f"{x:7.3f}ms"


def compare(control, mixed):
    cq, mq = control["query"], mixed["query"]
    ratio = None
    if cq["p50_ms"] and mq["p50_ms"] and cq["p50_ms"] > 0:
        ratio = mq["p50_ms"] / cq["p50_ms"]
    return {
        "query_p50_idle_ms": cq["p50_ms"],
        "query_p50_mixed_ms": mq["p50_ms"],
        "query_p95_idle_ms": cq["p95_ms"],
        "query_p95_mixed_ms": mq["p95_ms"],
        "query_p50_ratio": ratio,
        "qps_idle": cq["qps"],
        "qps_mixed": mq["qps"],
        "write_p50_ms": mixed["write"]["p50_ms"],
        "insert_p50_ms": mixed["write"]["insert_p50_ms"],
        "update_p50_ms": mixed["write"]["update_p50_ms"],
        "query_error_rate_mixed": mq["error_rate"],
        "write_error_rate_mixed": mixed["write"]["error_rate"],
        "writers_stall_readers": bool(
            ratio is not None and (ratio >= 1.4 or (mq["p95_ms"] and cq["p95_ms"] and mq["p95_ms"] >= 2.0 * cq["p95_ms"]))
        ),
    }


def start_server(args):
    cmd = [args.binary, "serve", "--port", str(args.port), "--ef", str(args.ef)]
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    server = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return server


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=8000, help="base corpus size (keep 2000–20000)")
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--port", type=int, default=8701)
    ap.add_argument("--seconds", type=float, default=8.0, help="duration of each phase")
    ap.add_argument("--query-workers", type=int, default=4)
    ap.add_argument("--insert-workers", type=int, default=1)
    ap.add_argument("--update-workers", type=int, default=1)
    ap.add_argument("--write-batch", type=int, default=1, help="docs per upsert (exclusive lock covers the batch)")
    ap.add_argument("--load-batch", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--workers", type=int, default=0, help="pass --workers to serve (0 = server default)")
    ap.add_argument("--binary", default="./zig-out/bin/openpuffer")
    ap.add_argument("--json-out", default="", help="optional path to write the result JSON")
    args = ap.parse_args()

    if args.n < 1 or args.n > 20000:
        sys.exit("--n must be 1–20000 (mixed R/W is a lock-contention bench, not a scale upsert storm)")
    if not os.path.exists(args.binary):
        sys.exit(f"missing binary {args.binary}; run zig build -Doptimize=ReleaseFast")

    rng = random.Random(7)
    print(f"generating {args.n} docs + 64 query vecs (dim {args.dim})...")
    docs = [unit_vec(args.dim, rng) for _ in range(args.n)]
    queries = [unit_vec(args.dim, rng) for _ in range(64)]

    ns = "mixed-bench"
    host = "127.0.0.1"
    base = f"http://{host}:{args.port}/v2/namespaces/{ns}"
    server = start_server(args)
    try:
        wait_ready(args.port, server)
        load_s = load_docs(base, docs, args.load_batch)
        print(f"loaded n={args.n} dim={args.dim} in {load_s:.2f}s")
        for qv in queries[: args.warmup]:
            http_json(f"{base}/query", query_payload(qv, args.k, args.ef), timeout=args.timeout)

        ids = IdSpace(args.n)
        control = run_window(
            "query-only control",
            args.seconds,
            host,
            args.port,
            ns,
            queries,
            args,
            ids,
            insert_n=0,
            update_n=0,
        )
        print_phase(control)

        mixed = run_window(
            f"mixed q={args.query_workers} ins={args.insert_workers} upd={args.update_workers} batch={args.write_batch}",
            args.seconds,
            host,
            args.port,
            ns,
            queries,
            args,
            ids,
            insert_n=args.insert_workers,
            update_n=args.update_workers,
        )
        print_phase(mixed)

        cmp = compare(control, mixed)
        print("=== mixed vs idle ===")
        if cmp["query_p50_ratio"] is not None:
            print(
                f"query p50 idle {fmt_ms(cmp['query_p50_idle_ms'])}  "
                f"mixed {fmt_ms(cmp['query_p50_mixed_ms'])}  "
                f"ratio {cmp['query_p50_ratio']:.2f}x"
            )
        else:
            print("query p50 unavailable")
        print(
            f"query p95 idle {fmt_ms(cmp['query_p95_idle_ms'])}  "
            f"mixed {fmt_ms(cmp['query_p95_mixed_ms'])}"
        )
        print(
            f"qps idle {cmp['qps_idle']:.1f}  mixed {cmp['qps_mixed']:.1f}  "
            f"write p50 {fmt_ms(cmp['write_p50_ms'])}  "
            f"insert p50 {fmt_ms(cmp['insert_p50_ms'])}  "
            f"update p50 {fmt_ms(cmp['update_p50_ms'])}"
        )
        print(
            f"errors query {cmp['query_error_rate_mixed']*100:.2f}%  "
            f"write {cmp['write_error_rate_mixed']*100:.2f}%"
        )
        if cmp["writers_stall_readers"]:
            print(
                "writers stall readers: YES — exclusive write lock around HNSW "
                "insert/update blocks lockShared query (expected)."
            )
        else:
            print(
                "writers stall readers: not obvious at p50/p95 on this modest corpus "
                "(insert still takes the exclusive lock for the whole HNSW mutation)."
            )

        out = {
            "n": args.n,
            "dim": args.dim,
            "k": args.k,
            "ef": args.ef,
            "seconds": args.seconds,
            "query_workers": args.query_workers,
            "insert_workers": args.insert_workers,
            "update_workers": args.update_workers,
            "write_batch": args.write_batch,
            "port": args.port,
            "load_s": load_s,
            "control": control,
            "mixed": mixed,
            "compare": cmp,
        }
        print("=== json ===")
        print(json.dumps(out, indent=2))
        if args.json_out:
            with open(args.json_out, "w") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
