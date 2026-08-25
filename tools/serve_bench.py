#!/usr/bin/env python3
"""Local HTTP serve-path bench (no API keys).

Starts `openpuffer serve`, upserts random unit vectors, then times ANN queries.

Modes:
  serial     one new TCP connection per query (urllib / qa_bench pattern)
  concurrent N threads, each opening a new connection
  keepalive  one HTTP/1.1 connection reused for all queries (http.client)
  keepalive+concurrency
             N persistent connections sharing the query list (real scale)

Usage:
    python3 tools/serve_bench.py [--n 20000] [--queries 400] [--concurrency 8]
    python3 tools/serve_bench.py --n 2000 --queries 400 --keepalive
    python3 tools/serve_bench.py --n 2000 --queries 400 --keepalive --concurrency 8
"""
from __future__ import annotations

import argparse
import http.client
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    s = sorted(samples)
    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return pct(0.50), pct(0.95), pct(0.99)


def query_new_conn(base, qv, k, timeout=60):
    t0 = time.perf_counter()
    r = http_json(f"{base}/query", {"rank_by": ["vector", "ANN", qv], "top_k": k}, timeout=timeout)
    return time.perf_counter() - t0, bool(r.get("rows"))


class KeepaliveClient:
    def __init__(self, host, port, ns):
        self.path = f"/v2/namespaces/{ns}/query"
        self.conn = http.client.HTTPConnection(host, port, timeout=60)

    def query(self, qv, k):
        body = json.dumps({"rank_by": ["vector", "ANN", qv], "top_k": k}).encode()
        t0 = time.perf_counter()
        self.conn.request("POST", self.path, body=body, headers={"content-type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        elapsed = time.perf_counter() - t0
        if resp.status != 200:
            raise RuntimeError(f"query status {resp.status}: {data[:200]!r}")
        r = json.loads(data)
        return elapsed, bool(r.get("rows"))

    def close(self):
        self.conn.close()


def run_keepalive_batch(host, port, ns, queries, k):
    client = KeepaliveClient(host, port, ns)
    samples = []
    hits = 0
    try:
        for qv in queries:
            elapsed, ok = client.query(qv, k)
            samples.append(elapsed)
            hits += int(ok)
    finally:
        client.close()
    return samples, hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--keepalive", action="store_true")
    ap.add_argument("--binary", default="./zig-out/bin/openpuffer")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"missing binary {args.binary}; run zig build -Doptimize=ReleaseFast")

    rng = random.Random(7)
    print(f"generating {args.n} docs + {args.queries} queries (dim {args.dim})...")
    docs = [unit_vec(args.dim, rng) for _ in range(args.n)]
    queries = [unit_vec(args.dim, rng) for _ in range(args.queries)]

    server = subprocess.Popen(
        [args.binary, "serve", "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    ns = "serve-bench"
    base = f"http://127.0.0.1:{args.port}/v2/namespaces/{ns}"
    try:
        for _ in range(100):
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{args.port}/v2/namespaces/", timeout=0.2
                )
                break
            except urllib.error.HTTPError:
                break
            except Exception:
                if server.poll() is not None:
                    sys.exit(f"server died:\n{server.stdout.read().decode()}")
                time.sleep(0.05)
        else:
            sys.exit("server did not become ready")

        t0 = time.perf_counter()
        B = 50
        for i in range(0, len(docs), B):
            chunk = docs[i : i + B]
            http_json(
                base,
                {
                    "upsert_columns": {
                        "id": list(range(i, i + len(chunk))),
                        "vector": chunk,
                    },
                    "distance_metric": "cosine_distance",
                },
            )
        load_s = time.perf_counter() - t0

        http_json(f"{base}/query", {"rank_by": ["vector", "ANN", queries[0]], "top_k": args.k})

        samples = []
        hits = 0
        wall0 = time.perf_counter()
        if args.keepalive and args.concurrency > 1:
            chunks = [queries[i :: args.concurrency] for i in range(args.concurrency)]
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = [
                    pool.submit(run_keepalive_batch, "127.0.0.1", args.port, ns, chunk, args.k)
                    for chunk in chunks
                    if chunk
                ]
                for fut in as_completed(futs):
                    part, ok = fut.result()
                    samples.extend(part)
                    hits += ok
            mode = f"keepalive concurrency={args.concurrency}"
        elif args.keepalive:
            client = KeepaliveClient("127.0.0.1", args.port, ns)
            try:
                for qv in queries:
                    elapsed, ok = client.query(qv, args.k)
                    samples.append(elapsed)
                    hits += int(ok)
            finally:
                client.close()
            mode = "keepalive"
        elif args.concurrency <= 1:
            for qv in queries:
                elapsed, ok = query_new_conn(base, qv, args.k)
                samples.append(elapsed)
                hits += int(ok)
            mode = "serial"
        else:
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                futs = [pool.submit(query_new_conn, base, qv, args.k) for qv in queries]
                for fut in as_completed(futs):
                    elapsed, ok = fut.result()
                    samples.append(elapsed)
                    hits += int(ok)
            mode = f"concurrency={args.concurrency}"
        wall = time.perf_counter() - wall0

        p50, p95, p99 = percentiles(samples)
        mean = statistics.mean(samples)
        qps = args.queries / wall if wall > 0 else 0
        print("=== openpuffer serve HTTP bench ===")
        print(
            f"mode={mode} n={args.n} queries={args.queries} dim={args.dim} k={args.k} "
            f"load={load_s:.2f}s wall={wall:.2f}s qps={qps:.1f}"
        )
        print(
            f"query p50 {p50*1000:7.3f}ms  p95 {p95*1000:7.3f}ms  p99 {p99*1000:7.3f}ms  "
            f"mean {mean*1000:7.3f}ms  nonempty {hits}/{args.queries}"
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=2)
        except subprocess.TimeoutExpired:
            server.kill()


if __name__ == "__main__":
    main()
