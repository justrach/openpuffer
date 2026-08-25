#!/usr/bin/env python3
"""1 vs N shard comparison at a modest n (default 8k, hard cap 20k).

Does not launch 1M / 2M single-process benches.

    python3 tools/shard_bench.py
    python3 tools/shard_bench.py --n 8000 --queries 80 --dim 1536 --ef 128

Measures, for the same total vectors:
  * direct 1-instance serve (no router)
  * router + 1 / 2 / 4 child serve processes
  * write-routing correctness (doc comes back only from the owning shard)
  * merge correctness (global top-k ≈ single-index top-k on a clustered set)
  * query p50 / QPS (keepalive serial + keepalive concurrency=4)

RSS is the sum of child ``openpuffer serve`` processes (router Python RSS
is listed separately and is not the scale problem).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_key import selftest as shard_key_selftest, shard_for
from shard_router import (
    ShardCluster,
    RouterState,
    merge_rows,
    rss_mib,
    serve_router,
    split_upsert,
)


def unit_vec(dim, rng):
    v = [rng.gauss(0.0, 1.0) for _ in range(dim)]
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def normalize(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def clustered_docs(n, dim, n_clusters, rng, noise=0.05):
    centroids = [unit_vec(dim, rng) for _ in range(n_clusters)]
    docs = []
    labels = []
    for i in range(n):
        c = centroids[i % n_clusters]
        v = [c[j] + rng.gauss(0.0, noise) for j in range(dim)]
        docs.append(normalize(v))
        labels.append(i % n_clusters)
    return docs, centroids, labels


def exact_topk(docs, qv, k):
    scored = []
    qn = normalize(list(qv))
    for i, v in enumerate(docs):
        dot = sum(a * b for a, b in zip(qn, v))
        scored.append((1.0 - dot, i))
    scored.sort()
    return [i for _, i in scored[:k]]


def overlap(a, b):
    sa, sb = set(a), set(b)
    if not sa:
        return 0.0
    return len(sa & sb) / len(sa)


def percentiles(samples):
    s = sorted(samples)
    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return pct(0.50), pct(0.95), pct(0.99)


def http_json(url, payload=None, method=None, timeout=60):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode()), r.status


class Keepalive:
    def __init__(self, host, port, ns):
        self.path = f"/v2/namespaces/{ns}/query"
        self.conn = HTTPConnection(host, port, timeout=60)
        self.conn.connect()
        if self.conn.sock is not None:
            try:
                self.conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

    def query(self, qv, k, ef=None):
        body = {"rank_by": ["vector", "ANN", qv], "top_k": k}
        if ef is not None:
            body["ef"] = ef
        raw = json.dumps(body).encode()
        t0 = time.perf_counter()
        self.conn.request("POST", self.path, body=raw, headers={"content-type": "application/json"})
        resp = self.conn.getresponse()
        data = resp.read()
        elapsed = time.perf_counter() - t0
        if resp.status != 200:
            raise RuntimeError(f"query status {resp.status}: {data[:200]!r}")
        return elapsed, json.loads(data)

    def close(self):
        self.conn.close()


def upsert_docs(base, docs, batch=40):
    t0 = time.perf_counter()
    for i in range(0, len(docs), batch):
        chunk = docs[i : i + batch]
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
    return time.perf_counter() - t0


def wait_port(port, proc=None, timeout=15.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            out = proc.stdout.read().decode() if proc.stdout else ""
            raise RuntimeError(f"process died:\n{out}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.15):
                return
        except OSError:
            time.sleep(0.05)
    raise TimeoutError(f"port {port} not ready")


class DirectServe:
    def __init__(self, binary, port, ef, workers):
        self.binary = binary
        self.port = port
        self.ef = ef
        self.workers = workers
        self.proc = None

    def start(self):
        cmd = [self.binary, "serve", "--port", str(self.port), "--ef", str(self.ef)]
        if self.workers is not None:
            cmd.extend(["--workers", str(self.workers)])
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        wait_port(self.port, self.proc)

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=2)

    def rss_total_mib(self):
        return rss_mib(self.proc.pid) if self.proc and self.proc.pid else 0.0

    @property
    def shard_ports(self):
        return [self.port]


class RoutedCluster:
    def __init__(self, binary, router_port, n_shards, shard_port_base, ef, workers, shard_by="doc"):
        self.router_port = router_port
        self.n_shards = n_shards
        ports = [shard_port_base + i for i in range(n_shards)]
        self.cluster = ShardCluster(binary, n_shards, ports, ef=ef, workers=workers, shard_by=shard_by)
        self.httpd = None
        self.thread = None
        self._router_pid = os.getpid()

    @property
    def port(self):
        return self.router_port

    @property
    def shard_ports(self):
        return self.cluster.shard_ports

    def start(self):
        self.cluster.start()
        state = RouterState(self.cluster)
        self.httpd = serve_router(state, self.router_port)
        self.thread = __import__("threading").Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        wait_port(self.router_port)

    def stop(self):
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        self.cluster.stop()

    def rss_total_mib(self):
        return self.cluster.rss_total_mib()

    def router_rss_mib(self):
        return rss_mib(self._router_pid)


def run_query_bench(port, ns, queries, k, ef, concurrency):
    samples = []
    hits = 0
    wall0 = time.perf_counter()
    if concurrency <= 1:
        c = Keepalive("127.0.0.1", port, ns)
        try:
            for qv in queries:
                elapsed, r = c.query(qv, k, ef)
                samples.append(elapsed)
                hits += int(bool(r.get("rows")))
        finally:
            c.close()
    else:
        chunks = [queries[i::concurrency] for i in range(concurrency)]

        def worker(chunk):
            client = Keepalive("127.0.0.1", port, ns)
            part, ok = [], 0
            try:
                for qv in chunk:
                    elapsed, r = client.query(qv, k, ef)
                    part.append(elapsed)
                    ok += int(bool(r.get("rows")))
            finally:
                client.close()
            return part, ok

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for part, ok in pool.map(worker, [c for c in chunks if c]):
                samples.extend(part)
                hits += ok
    wall = time.perf_counter() - wall0
    p50, p95, p99 = percentiles(samples)
    qps = len(queries) / wall if wall > 0 else 0
    return {
        "p50_ms": p50 * 1000,
        "p95_ms": p95 * 1000,
        "p99_ms": p99 * 1000,
        "mean_ms": statistics.mean(samples) * 1000,
        "qps": qps,
        "hits": hits,
        "n": len(queries),
    }


def check_routing(cluster_or_direct, ns, docs, n_shards, sample_ids):
    """Each sampled doc must live only on shard_for(ns, id, N)."""
    if n_shards <= 1:
        return {"ok": True, "checked": 0, "note": "single shard — routing is trivial"}
    ports = cluster_or_direct.shard_ports
    ok = 0
    failures = []
    for doc_id in sample_ids:
        owner = shard_for(ns, doc_id, n_shards)
        qv = docs[doc_id]
        body = json.dumps({"rank_by": ["vector", "ANN", qv], "top_k": 5}).encode()
        found_on = []
        for i, port in enumerate(ports):
            try:
                rows, status = _post_raw(port, f"/v2/namespaces/{ns}/query", body)
            except Exception as e:
                if i == owner:
                    failures.append({"id": doc_id, "err": str(e), "shard": i})
                continue
            ids = [r.get("id") for r in (rows.get("rows") or [])]
            if doc_id in ids:
                found_on.append(i)
        if found_on == [owner]:
            ok += 1
        else:
            failures.append({"id": doc_id, "owner": owner, "found_on": found_on})
    return {
        "ok": not failures,
        "checked": len(sample_ids),
        "passed": ok,
        "failures": failures[:8],
    }


def _post_raw(port, path, body):
    conn = HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        conn.request("POST", path, body=body, headers={"content-type": "application/json"})
        resp = conn.getresponse()
        data = resp.read()
        if resp.status == 404:
            return {"rows": []}, 404
        if resp.status != 200:
            raise RuntimeError(f"status {resp.status}: {data[:160]!r}")
        return json.loads(data), resp.status
    finally:
        conn.close()


def check_merge(single_port, sharded_port, ns_single, ns_shard, queries, docs, k, ef):
    """Router top-k vs single-index top-k vs exact, on the same clustered corpus."""
    overlaps_vs_single = []
    overlaps_vs_exact = []
    single_vs_exact = []
    for qv in queries:
        c1 = Keepalive("127.0.0.1", single_port, ns_single)
        cN = Keepalive("127.0.0.1", sharded_port, ns_shard)
        try:
            _, r1 = c1.query(qv, k, ef)
            _, rN = cN.query(qv, k, ef)
        finally:
            c1.close()
            cN.close()
        ids1 = [r["id"] for r in r1.get("rows") or []]
        idsN = [r["id"] for r in rN.get("rows") or []]
        exact = exact_topk(docs, qv, k)
        overlaps_vs_single.append(overlap(ids1, idsN))
        overlaps_vs_exact.append(overlap(exact, idsN))
        single_vs_exact.append(overlap(exact, ids1))
    return {
        "merge_vs_single": statistics.mean(overlaps_vs_single) if overlaps_vs_single else 0,
        "sharded_vs_exact": statistics.mean(overlaps_vs_exact) if overlaps_vs_exact else 0,
        "single_vs_exact": statistics.mean(single_vs_exact) if single_vs_exact else 0,
        "queries": len(queries),
    }


def bench_one(label, target, ns, docs, queries, k, ef, n_shards):
    base = f"http://127.0.0.1:{target.port}/v2/namespaces/{ns}"
    print(f"  upsert {len(docs)} docs onto {label}...", flush=True)
    load_s = upsert_docs(base, docs)
    stats, _ = http_json(base)
    if int(stats.get("count") or 0) != len(docs):
        raise RuntimeError(f"{label}: expected count {len(docs)}, got {stats}")

    sample = list(range(0, len(docs), max(1, len(docs) // 32)))[:32]
    routing = check_routing(target, ns, docs, n_shards, sample)

    # warmup
    http_json(f"{base}/query", {"rank_by": ["vector", "ANN", queries[0]], "top_k": k, "ef": ef})

    serial = run_query_bench(target.port, ns, queries, k, ef, concurrency=1)
    conc = run_query_bench(target.port, ns, queries, k, ef, concurrency=4)
    rss = target.rss_total_mib()
    return {
        "label": label,
        "shards": n_shards,
        "load_s": load_s,
        "count": stats.get("count"),
        "rss_mib": rss,
        "routing": routing,
        "serial": serial,
        "conc4": conc,
    }


def print_result(row):
    s, c = row["serial"], row["conc4"]
    route = row["routing"]
    route_s = "ok" if route.get("ok") else f"FAIL {route.get('failures')}"
    print(
        f"{row['label']:<14} shards={row['shards']}  load={row['load_s']:.2f}s  "
        f"rss={row['rss_mib']:.1f}MiB  count={row['count']}  route={route_s}"
    )
    print(
        f"  keepalive     p50 {s['p50_ms']:7.3f}ms  p95 {s['p95_ms']:7.3f}ms  "
        f"p99 {s['p99_ms']:7.3f}ms  qps {s['qps']:7.1f}  hits {s['hits']}/{s['n']}"
    )
    print(
        f"  ka conc=4     p50 {c['p50_ms']:7.3f}ms  p95 {c['p95_ms']:7.3f}ms  "
        f"p99 {c['p99_ms']:7.3f}ms  qps {c['qps']:7.1f}  hits {c['hits']}/{c['n']}"
    )


def unit_checks():
    shard_key_selftest()
    # merge: lower distance wins, ties break by id, de-dup
    merged = merge_rows(
        [[{"id": 1, "$distance": 0.2}, {"id": 2, "$distance": 0.4}],
         [{"id": 3, "$distance": 0.1}, {"id": 1, "$distance": 0.9}]],
        2,
    )
    assert [r["id"] for r in merged] == [3, 1]
    parts = split_upsert("serve-bench", json.dumps({
        "upsert_columns": {"id": [0, 1, 2, 3], "vector": [[0.0], [1.0], [2.0], [3.0]]},
        "distance_metric": "cosine_distance",
    }).encode(), 4, "doc")
    # fixtures from shard_key.selftest
    assert set(parts) == {0, 1, 2, 3}
    assert json.loads(parts[3])["upsert_columns"]["id"] == [0]
    assert json.loads(parts[0])["upsert_columns"]["id"] == [1]
    print("unit checks ok (shard_key + split + merge)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--queries", type=int, default=80)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--binary", default="./zig-out/bin/openpuffer")
    ap.add_argument("--workers", type=int, default=2, help="per serve process (2 avoids keep-alive pinning accept)")
    ap.add_argument("--skip-direct", action="store_true")
    args = ap.parse_args()

    if args.n > 20000:
        sys.exit("refuse n>20000 — this harness is the modest-n shard probe, not a 1M run")
    if not os.path.exists(args.binary):
        sys.exit(f"missing binary {args.binary}; run zig build -Doptimize=ReleaseFast")

    unit_checks()

    rng = random.Random(7)
    print(f"generating {args.n} random + 400 clustered docs (dim {args.dim})...", flush=True)
    docs = [unit_vec(args.dim, rng) for _ in range(args.n)]
    queries = [unit_vec(args.dim, rng) for _ in range(args.queries)]
    c_docs, centroids, _ = clustered_docs(400, args.dim, 4, random.Random(11))
    c_queries = [normalize([c[j] + 0.01 * random.Random(13 + i).gauss(0, 1) for j in range(args.dim)]) for i, c in enumerate(centroids)]

    configs = []
    if not args.skip_direct:
        configs.append(("1-direct", 1, True))
    configs.extend([("router-1", 1, False), ("router-2", 2, False), ("router-4", 4, False)])

    results = []
    merge_report = None
    single_merge_port = None
    single_merge_ns = None

    for label, n_shards, direct in configs:
        router_port = args.port + (0 if direct else 10 + n_shards)
        shard_base = router_port + 20
        ns = f"shard-bench-{label}"
        print(f"\n=== {label}  n={args.n} shards={n_shards} port={router_port} ===", flush=True)
        if direct:
            target = DirectServe(args.binary, router_port, args.ef, args.workers)
        else:
            target = RoutedCluster(
                args.binary, router_port, n_shards, shard_base, args.ef, args.workers
            )
        try:
            target.start()
            row = bench_one(label, target, ns, docs, queries, args.k, args.ef, n_shards)
            print_result(row)
            results.append(row)

            # clustered merge corpus on the same live target
            c_ns = f"{ns}-cluster"
            c_base = f"http://127.0.0.1:{target.port}/v2/namespaces/{c_ns}"
            upsert_docs(c_base, c_docs, batch=50)
            if label == "1-direct" or (label == "router-1" and single_merge_port is None):
                single_merge_port = target.port
                single_merge_ns = c_ns
                # hold this process until we finish merge vs router-4: run merge later
                # We need two live servers. Do merge online only when both exist.
                # Save clustered ids by querying now against this single and
                # comparing later is impossible once we stop it. So keep 1-direct
                # up... we stop in finally. Instead compute exact here and
                # compare sharded vs exact; plus vs-single when we still have
                # both (we don't). So: capture single-index top-k now.
                captured = []
                ck = Keepalive("127.0.0.1", target.port, c_ns)
                try:
                    for qv in c_queries:
                        _, r = ck.query(qv, args.k, args.ef)
                        captured.append([row["id"] for row in r.get("rows") or []])
                finally:
                    ck.close()
                merge_report = {"single_ids": captured, "exact": [exact_topk(c_docs, q, args.k) for q in c_queries]}
            if n_shards == 4 and merge_report is not None:
                ck = Keepalive("127.0.0.1", target.port, c_ns)
                try:
                    sharded_ids = []
                    for qv in c_queries:
                        _, r = ck.query(qv, args.k, args.ef)
                        sharded_ids.append([row["id"] for row in r.get("rows") or []])
                finally:
                    ck.close()
                vs_single = [overlap(a, b) for a, b in zip(merge_report["single_ids"], sharded_ids)]
                vs_exact = [overlap(a, b) for a, b in zip(merge_report["exact"], sharded_ids)]
                single_exact = [overlap(a, b) for a, b in zip(merge_report["exact"], merge_report["single_ids"])]
                merge_report["result"] = {
                    "merge_vs_single": statistics.mean(vs_single),
                    "sharded_vs_exact": statistics.mean(vs_exact),
                    "single_vs_exact": statistics.mean(single_exact),
                    "queries": len(c_queries),
                    "n_clustered": len(c_docs),
                    "example_single": merge_report["single_ids"][0],
                    "example_sharded": sharded_ids[0],
                    "example_exact": merge_report["exact"][0],
                }
                print(
                    f"  merge@k clustered n=400: vs-single={merge_report['result']['merge_vs_single']:.3f}  "
                    f"sharded-vs-exact={merge_report['result']['sharded_vs_exact']:.3f}  "
                    f"single-vs-exact={merge_report['result']['single_vs_exact']:.3f}"
                )
        finally:
            target.stop()
            time.sleep(0.2)

    print("\n=== comparison (same total vectors) ===")
    print(f"{'label':<14} {'shards':>6} {'p50 ka':>8} {'qps ka':>8} {'p50 c4':>8} {'qps c4':>8} {'rss':>8} route")
    for row in results:
        s, c = row["serial"], row["conc4"]
        print(
            f"{row['label']:<14} {row['shards']:>6} {s['p50_ms']:8.3f} {s['qps']:8.1f} "
            f"{c['p50_ms']:8.3f} {c['qps']:8.1f} {row['rss_mib']:8.1f} "
            f"{'ok' if row['routing'].get('ok') else 'FAIL'}"
        )
    if merge_report and merge_report.get("result"):
        print("merge:", json.dumps(merge_report["result"], indent=2))

    out_path = os.environ.get("SHARD_BENCH_JSON", "/tmp/openpuffer-shard-bench.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "n": args.n,
                "queries": args.queries,
                "dim": args.dim,
                "k": args.k,
                "ef": args.ef,
                "workers": args.workers,
                "results": results,
                "merge": merge_report.get("result") if merge_report else None,
            },
            f,
            indent=2,
        )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
