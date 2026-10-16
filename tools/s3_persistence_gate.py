#!/usr/bin/env python3
"""Persistence gate: openpuffer + tools/s3_fake.py, restarts under storage faults.

The invariant checked after every restart: each namespace is either back exactly
(same document count, same top-10 for fixed probes) or refused with 503 while
quarantined. It is never served partially, and nothing already persisted is lost.

Scenarios
  pagination      more objects than one LIST page; every namespace recovers
  write-faults    WAL/snapshot PUTs and DELETEs fail or drop mid-run; no write lost
  snapshot-fails  every compaction snapshot PUT fails: the WAL it would replace is kept,
                  nothing is lost, and compaction succeeds once the store recovers
  quarantine      one namespace's GETs fail at restart: 503 for it, the rest serve;
                  it heals in the background once the store recovers, no restart
  read-faults     60% of GETs fail at restart: no namespace served partially;
                  all heal; a final clean restart still matches
  list-fails      the bucket cannot be listed: the server refuses to start and
                  leaves the bucket untouched

Stdlib only. Usage:
  python3 tools/s3_persistence_gate.py [--bin zig-out/bin/openpuffer] [--latency-ms 2] [--keep]
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import math
import os
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

DIM = 32
TOOLS = os.path.dirname(os.path.abspath(__file__))


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def unit(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def user_vectors(user, n):
    """n documents in 8 tight clusters, plus 3 probe vectors (cluster centres)."""
    rng = random.Random(1000 + user)
    centres = [unit([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(8)]
    docs = [unit([c + rng.gauss(0, 0.15) for c in centres[i % 8]]) for i in range(n)]
    probes = [unit([c + rng.gauss(0, 0.05) for c in centres[i]]) for i in range(3)]
    return docs, probes


def stray_vector(rng):
    """Random direction: nearly orthogonal to every cluster, never in a probe's top 10."""
    return unit([rng.gauss(0, 1) for _ in range(DIM)])


class Client:
    """Keep-alive HTTP client, one connection per thread.

    Threads close their connection when their task ends: an idle keep-alive
    connection can hold one of openpuffer's Linux io_uring workers in a
    blocking recv, and with as many idle connections as workers it stalls."""

    def __init__(self, port):
        self.port = port
        self.local = threading.local()

    def close_thread(self):
        conn = getattr(self.local, "conn", None)
        if conn is not None:
            conn.close()
            self.local.conn = None

    def call(self, method, path, body=None):
        for attempt in range(2):
            conn = getattr(self.local, "conn", None)
            if conn is None:
                conn = self.local.conn = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
            try:
                payload = json.dumps(body).encode() if body is not None else None
                conn.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload else {})
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, (json.loads(data) if data else None)
            except (OSError, http.client.HTTPException):
                conn.close()
                self.local.conn = None
                if attempt:
                    raise

    def write(self, ns, ids, vecs, retries=3):
        """Upserts are idempotent, so a failed write is simply retried."""
        body = {"upsert_columns": {"id": ids, "vector": vecs}, "distance_metric": "cosine_distance"}
        for attempt in range(retries):
            status, _ = self.call("POST", f"/v2/namespaces/{ns}", body)
            if status == 200 or status == 503:
                return status, attempt
            time.sleep(0.2)
        return status, retries

    def top10(self, ns, vec):
        status, data = self.call("POST", f"/v2/namespaces/{ns}/query", {"rank_by": ["vector", "ANN", vec], "top_k": 10})
        return status, ([r["id"] for r in data["rows"]] if status == 200 else None)

    def count(self, ns):
        status, data = self.call("GET", f"/v2/namespaces/{ns}")
        return status, (data or {}).get("count")


class Rig:
    def __init__(self, binary, latency_ms, workdir):
        self.binary, self.workdir = binary, workdir
        self.bucket = os.path.join(workdir, "bucket")
        self.s3_port, self.port = free_port(), free_port()
        self.s3 = subprocess.Popen([sys.executable, os.path.join(TOOLS, "s3_fake.py"), "--port", str(self.s3_port),
                                    "--dir", self.bucket, "--latency-ms", str(latency_ms)],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.server = None
        self.log_path = None
        self.client = Client(self.port)
        self.wait(lambda: self.s3_stats() is not None, 10, "fake S3 did not start")

    # fake S3 control
    def s3_call(self, method, path, body=None):
        try:
            conn = http.client.HTTPConnection("127.0.0.1", self.s3_port, timeout=5)
            conn.request(method, path, body=json.dumps(body).encode() if body is not None else None)
            return json.loads(conn.getresponse().read())
        except OSError:
            return None

    def s3_stats(self):
        return self.s3_call("GET", "/__stats")

    def faults(self, **kw):
        base = {"get_503_pct": 0, "put_503_pct": 0, "delete_503_pct": 0, "drop_pct": 0,
                "fail_prefix": "", "fail_put_suffix": "", "list_503": False}
        self.s3_call("POST", "/__faults", {**base, **kw})

    # openpuffer lifecycle
    def start(self, label, expect_ready=True, timeout=120):
        self.log_path = os.path.join(self.workdir, f"openpuffer-{label}.log")
        env = {**os.environ, "AWS_ACCESS_KEY_ID": "gate", "AWS_SECRET_ACCESS_KEY": "gate",
               "AWS_REGION": "us-east-1", "OPENPUFFER_HEAL_INITIAL_MS": "300"}
        self.server = subprocess.Popen([self.binary, "serve", "--port", str(self.port), "--s3-bucket", "gate",
                                        "--s3-endpoint", f"http://127.0.0.1:{self.s3_port}"],
                                       stdout=open(self.log_path, "w"), stderr=subprocess.STDOUT, env=env)
        if not expect_ready:
            return
        self.wait(lambda: "openpuffer serving" in self.log() or self.server.poll() is not None, timeout,
                  "openpuffer did not start")
        if self.server.poll() is not None:
            raise AssertionError(f"openpuffer exited during startup:\n{self.log()[-2000:]}")
        self.client = Client(self.port)

    def stop(self):
        if self.server and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(10)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()
        self.server = None

    def close(self):
        self.stop()
        self.s3.terminate()
        self.s3.wait()

    def log(self):
        try:
            with open(self.log_path) as f:
                return f.read()
        except OSError:
            return ""

    @staticmethod
    def wait(cond, timeout, message):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if cond():
                return
            time.sleep(0.1)
        raise AssertionError(message)

    def bucket_files(self):
        out = {}
        for d, _, files in os.walk(self.bucket):
            for f in files:
                p = os.path.join(d, f)
                out[os.path.relpath(p, self.bucket)] = os.path.getmtime(p)
        return out


class Gate:
    def __init__(self, rig, users, docs, batch):
        self.rig, self.users, self.docs, self.batch = rig, users, docs, batch
        self.expected = {}   # ns -> expected document count
        self.probes = {}     # ns -> [(vector, top10)]
        self.stray = random.Random(99)
        self.next_stray = 9_000_000

    def ns(self, u):
        return f"u{u}"

    def load(self):
        """Every user writes its documents in batches, concurrently."""
        retried = 0

        def one(u):
            try:
                load_user(u)
            finally:
                self.rig.client.close_thread()

        def load_user(u):
            nonlocal retried
            docs, probes = user_vectors(u, self.docs)
            for start in range(0, self.docs, self.batch):
                ids = list(range(start, min(start + self.batch, self.docs)))
                status, attempts = self.rig.client.write(self.ns(u), ids, docs[start:start + self.batch])
                assert status == 200, f"{self.ns(u)}: write failed with {status} after retries"
                retried += attempts
            self.expected[self.ns(u)] = self.docs
            self.probes[self.ns(u)] = [(p, self.rig.client.top10(self.ns(u), p)[1]) for p in probes]

        self.rig.client.close_thread()
        with cf.ThreadPoolExecutor(8) as pool:
            list(pool.map(one, range(self.users)))
        return retried

    def write_strays(self, per_user):
        """Extra single-doc writes that never change probe results. Returns refused namespaces."""
        refused = set()
        for u in range(self.users):
            for _ in range(per_user):
                self.next_stray += 1
                status, _ = self.rig.client.write(self.ns(u), [self.next_stray], [stray_vector(self.stray)])
                if status == 200:
                    self.expected[self.ns(u)] += 1
                else:
                    refused.add(self.ns(u))
        return refused

    def check(self, allow_quarantined=False):
        """Every namespace is exact, or (if allowed) refused with 503. Returns the refused set."""
        refused = set()
        for u in range(self.users):
            ns = self.ns(u)
            status, count = self.rig.client.count(ns)
            if status == 503 and allow_quarantined:
                q_status, _ = self.rig.client.top10(ns, self.probes[ns][0][0])
                assert q_status == 503, f"{ns}: count says quarantined but query answered {q_status}"
                refused.add(ns)
                continue
            assert status == 200, f"{ns}: stats answered {status}"
            assert count == self.expected[ns], f"{ns}: {count} documents, expected {self.expected[ns]}"
            for vec, want in self.probes[ns]:
                q_status, got = self.rig.client.top10(ns, vec)
                assert q_status == 200 and got == want, f"{ns}: probe top-10 changed"
        return refused


def run(args):
    workdir = tempfile.mkdtemp(prefix="openpuffer-s3-gate-")
    rig = Rig(os.path.abspath(args.bin), args.latency_ms, workdir)
    results = []

    def step(name, fn):
        t = time.time()
        detail = fn()
        results.append((name, time.time() - t, detail))
        print(f"PASS {name:<13} {time.time() - t:5.1f}s  {detail}", flush=True)

    try:
        gate = Gate(rig, users=args.users, docs=args.docs, batch=args.batch)

        def pagination():
            rig.s3_call("POST", "/__faults", {"page_size": 50})
            rig.start("load")
            gate.load()
            objects = len(rig.bucket_files())
            assert objects > 50, f"only {objects} objects; the bucket must span several LIST pages"
            rig.stop()
            rig.start("pagination")
            gate.check()
            return f"{gate.users} namespaces from {objects} objects in pages of 50"

        def write_faults():
            before = rig.s3_stats()
            rig.faults(put_503_pct=20, delete_503_pct=20, drop_pct=5)
            more = Gate(rig, users=gate.users, docs=gate.docs, batch=gate.batch // 2)
            more.ns = lambda u: f"w{u}"
            retried = more.load()
            rig.faults()
            after = rig.s3_stats()
            requests = sum(after[k] - before[k] for k in ("PUT", "GET", "LIST", "DELETE")) + \
                (after["injected_503"] - before["injected_503"]) + (after["dropped"] - before["dropped"])
            conns = after["connections"] - before["connections"]
            assert conns * 4 < requests, f"{conns} connections for {requests} requests: connections are not reused"
            rig.stop()
            rig.start("after-write-faults")
            gate.check()
            more.check()
            gate.expected.update(more.expected)
            gate.probes.update(more.probes)
            gate.users_extra = more
            return (f"{after['injected_503'] - before['injected_503']} injected 503s, "
                    f"{after['dropped'] - before['dropped']} dropped connections, {retried} client-level retries; "
                    f"{conns} S3 connections for {requests} requests")

        def snapshot_fails():
            # 12 writes per namespace: compaction is due after 8, and every snapshot PUT fails
            rig.faults(fail_put_suffix="snapshot.bin")
            snap = Gate(rig, users=4, docs=120, batch=10)
            snap.ns = lambda u: f"s{u}"
            snap.load()
            wal = sum(1 for k in rig.bucket_files() if k.startswith("gate/openpuffer/s") and "/wal/" in k)
            assert wal >= 4 * 12, f"only {wal} WAL segments left for 4 namespaces x 12 writes: WAL was deleted"
            assert "keeping its WAL" in rig.log(), "compaction was never attempted"
            rig.faults()
            snap.write_strays(1)  # one more flush each: the re-armed compaction now succeeds
            rig.stop()
            rig.start("after-snapshot-fails")
            snap.check()
            snaps = [k for k in rig.bucket_files() if k.startswith("gate/openpuffer/s") and k.endswith("snapshot.bin")]
            assert len(snaps) == 4, f"{len(snaps)} snapshots after the store recovered"
            gate.expected.update(snap.expected)
            gate.probes.update(snap.probes)
            return f"WAL kept through failed snapshots ({wal} segments), all 4 namespaces exact after restart"

        def quarantine():
            rig.stop()
            rig.faults(fail_prefix="openpuffer/u3/")
            rig.start("quarantine")
            refused = gate.check(allow_quarantined=True)
            assert refused == {"u3"}, f"expected only u3 quarantined, got {sorted(refused)}"
            status, _ = rig.client.write("u3", [1], [stray_vector(random.Random(1))])
            assert status == 503, f"a write to a quarantined namespace answered {status}"
            rig.faults()
            rig.wait(lambda: rig.client.count("u3")[0] == 200, 30, "u3 did not heal")
            gate.check()
            assert "healed 'u3'" in rig.log(), "no heal logged"
            return "u3 refused reads and writes (503) while its GETs failed, then healed without a restart"

        def read_faults():
            rig.stop()
            # 60%: most requests still succeed within 5 attempts, a few namespaces exhaust them
            rig.faults(get_503_pct=60)
            rig.start("read-faults")
            refused = gate.check(allow_quarantined=True)
            refused_writes = gate.write_strays(2)
            assert refused_writes == refused, "a quarantined namespace accepted a write, or a served one refused"
            extra = getattr(gate, "users_extra", None)
            if extra:
                refused |= extra.check(allow_quarantined=True)
            rig.faults()
            names = [gate.ns(u) for u in range(gate.users)] + ([extra.ns(u) for u in range(extra.users)] if extra else [])
            rig.wait(lambda: all(rig.client.count(n)[0] == 200 for n in names), 60,
                     "quarantined namespaces did not heal")
            gate.check()
            if extra:
                extra.check()
            rig.stop()
            rig.start("clean")
            gate.check()
            if extra:
                extra.check()
            injected = rig.s3_stats()["injected_503"]
            return (f"{len(refused)} namespace(s) quarantined at restart and healed; none served partially; "
                    f"{injected} total injected 503s")

        def list_fails():
            rig.stop()
            files = rig.bucket_files()
            rig.faults(list_503=True)
            rig.start("list-fails", expect_ready=False)
            try:
                code = rig.server.wait(60)
            except subprocess.TimeoutExpired:
                raise AssertionError("openpuffer kept running with an unlistable bucket")
            assert code != 0, "openpuffer exited 0 with an unlistable bucket"
            assert "openpuffer serving" not in rig.log(), "openpuffer served with an unlistable bucket"
            assert rig.bucket_files() == files, "the bucket changed"
            rig.faults()
            return f"refused to start (exit {code}); bucket untouched"

        step("pagination", pagination)
        step("write-faults", write_faults)
        step("snapshot-fails", snapshot_fails)
        step("quarantine", quarantine)
        step("read-faults", read_faults)
        step("list-fails", list_fails)
        print(f"all {len(results)} persistence scenarios passed")
        return 0
    except AssertionError as e:
        print(f"FAIL: {e}", flush=True)
        log = rig.log()
        if log:
            print("---- openpuffer log (tail) ----")
            print(log[-3000:])
        return 1
    finally:
        rig.close()
        if args.keep:
            print(f"kept {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", default="zig-out/bin/openpuffer")
    p.add_argument("--latency-ms", type=float, default=2, help="fake S3 latency per request")
    p.add_argument("--users", type=int, default=24)
    p.add_argument("--docs", type=int, default=100)
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--keep", action="store_true", help="keep the bucket and logs")
    sys.exit(run(p.parse_args()))


if __name__ == "__main__":
    main()
