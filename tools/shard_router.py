#!/usr/bin/env python3
"""Python fallback shard router in front of N ``openpuffer serve`` processes.

Prefer the Zig/io_uring path (keep-alive to children):

    ./zig-out/bin/openpuffer shard-router --shards 4 --port 8800 --ef 128

This Python process is the fallback when that binary is unavailable.
Each child holds ~1/N of the documents and has its own HNSW graph.

    python3 tools/shard_router.py --shards 4 --port 8800
    OPENPUFFER_SHARDS=4 python3 tools/shard_router.py --port 8800

Process model
-------------
* N child ``openpuffer serve --port P_i --ef 128 --workers W`` processes.
* One Python router on ``--port`` (default 8800).
* Child ports default to ``--port + 1 .. + N`` (override with
  ``--shard-port-base``).

Routing
-------
* Writes (POST /v2/namespaces/{ns}): split the upsert batch by
  ``shard_key.shard_for`` and send each slice to exactly one child.
  A document is never written to two shards.
* Queries (POST .../query): scatter-gather. Fan the same ANN body to
  every child that might own the namespace (all children in ``doc``
  mode; one child in ``namespace`` mode). Merge rows by ``$distance``
  ascending and keep the requested top_k.
* GET stats: sum ``count`` across children; dim is the first non-zero.
* DELETE: fan out; ignore per-child 404.

Limits (honest)
---------------
* Cross-shard recall is a merge of *per-shard ANN* results, not a
  unified HNSW walk. If a shard's local ANN misses a true neighbor
  that lives on that shard, the merge misses it too.
* Every query pays a localhost hop + JSON encode/decode + merge.
* On a 4-core host, 4 shards × ncpu workers oversubscribe. Default
  child ``--workers`` is 1 when shards > 1.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import socket
import subprocess
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_key import SHARD_BY_DOC, SHARD_BY_NAMESPACE, shard_for

PREFIX = "/v2/namespaces/"


def parse_top_k(body: bytes) -> int:
    try:
        obj = json.loads(body)
    except json.JSONDecodeError:
        return 10
    for key in ("top_k", "limit"):
        v = obj.get(key)
        if isinstance(v, int) and v > 0:
            return v
    return 10


def split_upsert(namespace: str, body: bytes, n_shards: int, shard_by: str):
    """Split a turbopuffer write body into per-shard payloads.

    Returns ``{shard_idx: json_bytes}``. Empty shards are omitted.
    Distance-metric and unknown top-level keys are copied onto every slice.
    """
    obj = json.loads(body)
    extra = {k: v for k, v in obj.items() if k not in ("upsert_columns", "upsert_rows")}
    buckets: dict[int, list] = {i: [] for i in range(n_shards)}

    if "upsert_columns" in obj:
        cols = obj["upsert_columns"]
        if not isinstance(cols, dict):
            raise ValueError("upsert_columns must be object")
        ids = cols.get("id")
        vecs = cols.get("vector")
        if not isinstance(ids, list) or not isinstance(vecs, list):
            raise ValueError("id/vector columns must be arrays")
        if len(ids) != len(vecs):
            raise ValueError("id/vector column length mismatch")
        other_cols = {k: v for k, v in cols.items() if k not in ("id", "vector")}
        for i, (doc_id, vec) in enumerate(zip(ids, vecs)):
            sid = shard_for(namespace, doc_id, n_shards, shard_by)
            row = {"id": doc_id, "vector": vec, "_i": i}
            buckets[sid].append(row)
        out = {}
        for sid, rows in buckets.items():
            if not rows:
                continue
            payload = dict(extra)
            col = {
                "id": [r["id"] for r in rows],
                "vector": [r["vector"] for r in rows],
            }
            for k, arr in other_cols.items():
                if isinstance(arr, list):
                    col[k] = [arr[r["_i"]] for r in rows]
            payload["upsert_columns"] = col
            out[sid] = json.dumps(payload).encode()
        return out

    if "upsert_rows" in obj:
        rows = obj["upsert_rows"]
        if not isinstance(rows, list):
            raise ValueError("upsert_rows must be array")
        for row in rows:
            if not isinstance(row, dict) or "id" not in row:
                raise ValueError("row missing id")
            sid = shard_for(namespace, row["id"], n_shards, shard_by)
            buckets[sid].append(row)
        out = {}
        for sid, rows_s in buckets.items():
            if not rows_s:
                continue
            payload = dict(extra)
            payload["upsert_rows"] = rows_s
            out[sid] = json.dumps(payload).encode()
        return out

    raise ValueError("missing upsert_columns/upsert_rows")


def merge_rows(results: list[list[dict]], top_k: int) -> list[dict]:
    """Merge per-shard ``rows`` by ``$distance`` ascending (cosine)."""
    merged: list[dict] = []
    for rows in results:
        for r in rows:
            if isinstance(r, dict) and "id" in r:
                merged.append(r)
    merged.sort(key=lambda r: (float(r.get("$distance", 1e9)), r.get("id", 0)))
    # de-dup by id (should not happen if writes are exclusive)
    seen = set()
    out = []
    for r in merged:
        rid = r["id"]
        if rid in seen:
            continue
        seen.add(rid)
        out.append(r)
        if len(out) >= top_k:
            break
    return out


def rss_mib(pid: int) -> float | None:
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        return None
    return None


def _nodelay(sock: socket.socket) -> None:
    try:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    except OSError:
        pass


class ShardClient:
    """Fresh TCP_NODELAY connection per child request.

    Keep-alive reuse against ``openpuffer serve`` raced under scatter-gather
    (broken pipe / accept-thread pin). A new localhost connection is the
    honest first-experiment hop; pooling is a follow-up.
    """

    def __init__(self, ports: list[int], timeout: float = 60.0):
        self.ports = ports
        self.timeout = timeout

    def request(self, idx: int, method: str, path: str, body: bytes | None = None):
        headers = {"connection": "close"}
        if body is not None:
            headers["content-type"] = "application/json"
            headers["content-length"] = str(len(body))
        last_err = None
        for _attempt in range(3):
            conn = HTTPConnection("127.0.0.1", self.ports[idx], timeout=self.timeout)
            try:
                conn.connect()
                if conn.sock is not None:
                    _nodelay(conn.sock)
                conn.request(method, path, body=body, headers=headers)
                resp = conn.getresponse()
                data = resp.read()
                return resp.status, data
            except (HTTPException, OSError, ConnectionError) as e:
                last_err = e
            finally:
                try:
                    conn.close()
                except Exception:
                    pass
        raise last_err  # pragma: no cover


class ShardCluster:
    def __init__(
        self,
        binary: str,
        n_shards: int,
        shard_ports: list[int],
        ef: int = 128,
        workers: int | None = None,
        shard_by: str = SHARD_BY_DOC,
    ):
        self.binary = binary
        self.n_shards = n_shards
        self.shard_ports = shard_ports
        self.ef = ef
        # workers=1 pins the io_uring accept thread on a keep-alive socket
        # (see E024 / mixed-workload notes). Two workers leave accept free
        # when the router holds a persistent child connection.
        self.workers = workers if workers is not None else (2 if n_shards > 1 else 1)
        self.shard_by = shard_by
        self.procs: list[subprocess.Popen] = []
        self.client = ShardClient(shard_ports)
        self._fanout_pool = ThreadPoolExecutor(max_workers=max(1, n_shards))

    def start(self) -> None:
        if not os.path.exists(self.binary):
            raise FileNotFoundError(
                f"missing binary {self.binary}; run zig build -Doptimize=ReleaseFast"
            )
        for i, port in enumerate(self.shard_ports):
            cmd = [self.binary, "serve", "--port", str(port), "--ef", str(self.ef)]
            if self.workers is not None:
                cmd.extend(["--workers", str(self.workers)])
            logf = open(os.devnull, "wb")
            proc = subprocess.Popen(
                cmd,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
            self.procs.append(proc)
            print(
                f"shard[{i}] pid={proc.pid} port={port} workers={self.workers or 'ncpu'}",
                flush=True,
            )
        self.wait_ready()

    def wait_ready(self, timeout_s: float = 15.0) -> None:
        deadline = time.time() + timeout_s
        pending = set(self.shard_ports)
        while pending and time.time() < deadline:
            for i, proc in enumerate(self.procs):
                if proc.poll() is not None:
                    out = proc.stdout.read().decode() if proc.stdout else ""
                    raise RuntimeError(f"shard[{i}] died:\n{out}")
            still = set()
            for port in pending:
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=0.15):
                        pass
                except OSError:
                    still.add(port)
            pending = still
            if pending:
                time.sleep(0.05)
        if pending:
            raise TimeoutError(f"shards not ready on ports {sorted(pending)}")

    def stop(self) -> None:
        self._fanout_pool.shutdown(wait=False, cancel_futures=True)
        for proc in self.procs:
            if proc.poll() is None:
                proc.terminate()
        for proc in self.procs:
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=2)
        self.procs.clear()

    def rss_total_mib(self) -> float:
        total = 0.0
        for proc in self.procs:
            if proc.pid is None:
                continue
            m = rss_mib(proc.pid)
            if m is not None:
                total += m
        return total

    def query_targets(self, namespace: str) -> list[int]:
        if self.shard_by == SHARD_BY_NAMESPACE:
            return [shard_for(namespace, 0, self.n_shards, SHARD_BY_NAMESPACE)]
        return list(range(self.n_shards))

    def fanout(self, method: str, path: str, body: bytes | None, targets: list[int]):
        if len(targets) == 1:
            status, data = self.client.request(targets[0], method, path, body)
            return [(targets[0], status, data)]

        futs = {
            self._fanout_pool.submit(self.client.request, i, method, path, body): i
            for i in targets
        }
        out = []
        for fut in as_completed(futs):
            idx = futs[fut]
            status, data = fut.result()
            out.append((idx, status, data))
        out.sort(key=lambda t: t[0])
        return out


class RouterState:
    def __init__(self, cluster: ShardCluster):
        self.cluster = cluster


def handle_route(state: RouterState, method: str, path: str, body: bytes) -> tuple[int, bytes]:
    cluster = state.cluster
    if path in ("/health", "/"):
        kids = []
        for i, proc in enumerate(cluster.procs):
            kids.append(
                {
                    "index": i,
                    "port": cluster.shard_ports[i],
                    "pid": proc.pid,
                    "alive": proc.poll() is None,
                    "rss_mib": rss_mib(proc.pid) if proc.pid else None,
                }
            )
        payload = {
            "ok": True,
            "shards": cluster.n_shards,
            "shard_by": cluster.shard_by,
            "hash": "fnv1a64(namespace + NUL + decimal_id) % N  (doc mode)",
            "children": kids,
            "rss_mib": cluster.rss_total_mib(),
        }
        return 200, json.dumps(payload).encode()

    if not path.startswith(PREFIX):
        return 404, b'{"error":"unknown route"}'

    rest = unquote(path[len(PREFIX) :])
    method = method.upper()

    if rest.endswith("/query") and method == "POST":
        ns = rest[: -len("/query")]
        top_k = parse_top_k(body)
        targets = cluster.query_targets(ns)
        replies = cluster.fanout("POST", f"{PREFIX}{ns}/query", body, targets)
        rows = []
        errors = []
        any_ok = False
        for idx, status, data in replies:
            if status == 404:
                continue
            if status != 200:
                errors.append({"shard": idx, "status": status, "body": data[:200].decode("utf-8", "replace")})
                continue
            any_ok = True
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                errors.append({"shard": idx, "status": status, "body": "invalid json"})
                continue
            rows.append(obj.get("rows") or [])
        if errors and not any_ok:
            return 502, json.dumps({"error": "all shards failed", "details": errors}).encode()
        if not any_ok:
            return 404, b'{"error":"namespace not found"}'
        merged = merge_rows(rows, top_k)
        return 200, json.dumps({"rows": merged, "usage": {}}).encode()

    if rest.endswith("/snapshot") and method == "POST":
        ns = rest[: -len("/snapshot")]
        replies = cluster.fanout("POST", f"{PREFIX}{ns}/snapshot", body or b"", list(range(cluster.n_shards)))
        return 200, json.dumps({"ok": True, "shards": [{"index": i, "status": s} for i, s, _ in replies]}).encode()

    if method == "POST":
        ns = rest
        try:
            slices = split_upsert(ns, body, cluster.n_shards, cluster.shard_by)
        except (ValueError, json.JSONDecodeError) as e:
            return 400, json.dumps({"error": str(e)}).encode()
        if not slices:
            return 200, b'{"ok":true}'

        def one(item):
            sid, payload = item
            return sid, cluster.client.request(sid, "POST", f"{PREFIX}{ns}", payload)

        results = []
        items = list(slices.items())
        if len(items) == 1:
            results.append(one(items[0]))
        else:
            futs = [cluster._fanout_pool.submit(one, item) for item in items]
            for fut in as_completed(futs):
                results.append(fut.result())
        for sid, (status, data) in results:
            if status != 200:
                return status, data or json.dumps({"error": "shard write failed", "shard": sid}).encode()
        return 200, b'{"ok":true}'

    if method == "GET":
        ns = rest
        replies = cluster.fanout("GET", f"{PREFIX}{ns}", None, cluster.query_targets(ns))
        count = 0
        dim = 0
        any_ok = False
        for _idx, status, data in replies:
            if status == 404:
                continue
            if status != 200:
                return status, data
            any_ok = True
            obj = json.loads(data)
            count += int(obj.get("count") or 0)
            if not dim:
                dim = int(obj.get("dim") or 0)
        if not any_ok:
            return 404, b'{"error":"namespace not found"}'
        return 200, json.dumps({"namespace": ns, "dim": dim, "count": count}).encode()

    if method == "DELETE":
        ns = rest
        replies = cluster.fanout("DELETE", f"{PREFIX}{ns}", None, list(range(cluster.n_shards)))
        return 200, json.dumps({"ok": True, "shards": len(replies)}).encode()

    return 405, b'{"error":"method not allowed"}'


class make_handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def __init__(self, state: RouterState, *args, **kwargs):
        self.state = state
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):
        # keep router quiet during benches; children still log startup
        return

    def _read_body(self) -> bytes:
        n = int(self.headers.get("Content-Length") or 0)
        if n <= 0:
            return b""
        return self.rfile.read(n)

    def _send(self, status: int, body: bytes) -> None:
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("connection", "keep-alive")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, method: str) -> None:
        try:
            body = self._read_body()
            status, out = handle_route(self.state, method, self.path, body)
            self._send(status, out)
        except BrokenPipeError:
            return
        except Exception:
            traceback.print_exc()
            try:
                self._send(500, b'{"error":"internal"}')
            except (BrokenPipeError, OSError):
                return

    def do_GET(self):
        self._dispatch("GET")

    def do_POST(self):
        self._dispatch("POST")

    def do_DELETE(self):
        self._dispatch("DELETE")


def serve_router(state: RouterState, port: int) -> ThreadingHTTPServer:
    def factory(*args, **kwargs):
        return make_handler(state, *args, **kwargs)

    class NodelayServer(ThreadingHTTPServer):
        daemon_threads = True
        allow_reuse_address = True

        def get_request(self):
            sock, addr = super().get_request()
            _nodelay(sock)
            return sock, addr

    httpd = NodelayServer(("127.0.0.1", port), factory)
    return httpd


def resolve_shards(cli_value: int | None) -> int:
    if cli_value is not None:
        return cli_value
    env = os.environ.get("OPENPUFFER_SHARDS")
    if env:
        return max(1, int(env))
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="openpuffer multi-instance shard router")
    ap.add_argument("--shards", type=int, default=None, help="N child serve processes (or OPENPUFFER_SHARDS)")
    ap.add_argument("--port", type=int, default=8800, help="router listen port")
    ap.add_argument("--shard-port-base", type=int, default=None, help="first child port (default: --port+1)")
    ap.add_argument("--ef", type=int, default=128)
    ap.add_argument("--workers", type=int, default=None, help="per-child --workers (default: 1 if shards>1)")
    ap.add_argument("--shard-by", choices=(SHARD_BY_DOC, SHARD_BY_NAMESPACE), default=SHARD_BY_DOC)
    ap.add_argument("--binary", default="./zig-out/bin/openpuffer")
    args = ap.parse_args(argv)

    n = resolve_shards(args.shards)
    if n < 1:
        sys.exit("--shards must be >= 1")
    base = args.shard_port_base if args.shard_port_base is not None else args.port + 1
    ports = [base + i for i in range(n)]
    if args.port in ports:
        sys.exit("router port collides with a shard port")

    cluster = ShardCluster(
        binary=args.binary,
        n_shards=n,
        shard_ports=ports,
        ef=args.ef,
        workers=args.workers,
        shard_by=args.shard_by,
    )
    cluster.start()
    state = RouterState(cluster)
    httpd = serve_router(state, args.port)
    print(
        f"openpuffer shard router on http://127.0.0.1:{args.port}  "
        f"shards={n} by={args.shard_by} children={ports}",
        flush=True,
    )
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        httpd.server_close()
        cluster.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
