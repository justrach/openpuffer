#!/usr/bin/env python3
"""Local HTTP serve-path bench (no API keys).

Starts `openpuffer serve`, upserts random unit vectors, then times ANN queries
over localhost HTTP — the same access pattern as tools/qa_bench.py (one new
TCP connection per request via urllib).

Usage:
    python3 tools/serve_bench.py [--n 2000] [--queries 200] [--dim 1536] [--k 10] [--ef-note 256]
"""
from __future__ import annotations

import argparse
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


def http_json(url, payload=None, method=None, timeout=30):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--binary", default="./zig-out/bin/openpuffer")
    args = ap.parse_args()

    if not os.path.exists(args.binary):
        sys.exit(f"missing binary {args.binary}; run zig build -Doptimize=ReleaseFast")

    rng = random.Random(7)
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
                break  # process is accepting; 404 on empty ns name is expected
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

        # warmup
        http_json(f"{base}/query", {"rank_by": ["vector", "ANN", queries[0]], "top_k": args.k})

        samples = []
        hits = 0
        for qv in queries:
            t0 = time.perf_counter()
            r = http_json(f"{base}/query", {"rank_by": ["vector", "ANN", qv], "top_k": args.k})
            samples.append(time.perf_counter() - t0)
            rows = r.get("rows", [])
            hits += 1 if rows else 0

        p50, p95, p99 = percentiles(samples)
        mean = statistics.mean(samples)
        print("=== openpuffer serve HTTP bench ===")
        print(f"n={args.n} queries={args.queries} dim={args.dim} k={args.k} load={load_s:.2f}s")
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
