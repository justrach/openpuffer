#!/usr/bin/env python3
"""Keep-alive check: idle client connections must not stall the server.

Runs `openpuffer serve --workers 2` and checks, with more idle keep-alive
connections open than there are workers:

  idle-clients    a new client is answered promptly
  parked-reuse    every idle connection still works when it speaks again
  pipelined       two requests sent in one write both get answers
  busy-clients    32 clients x 20 requests each all succeed, with the idle ones still open
  fd-release      closing the idle connections releases their descriptors (Linux)

Stdlib only. Usage: python3 tools/keepalive_check.py [--bin zig-out/bin/openpuffer]
"""
import argparse
import concurrent.futures as cf
import http.client
import json
import os
import socket
import subprocess
import sys
import time

IDLE = 6          # idle keep-alive connections; more than the 2 workers
TIMEOUT = 3.0     # a stalled server answers nothing within this


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def request(conn, method, path, body=None):
    payload = json.dumps(body) if body is not None else None
    conn.request(method, path, body=payload, headers={"Content-Type": "application/json"} if payload else {})
    resp = conn.getresponse()
    return resp.status, resp.read()


def open_fds(pid):
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except OSError:
        return None  # not Linux


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bin", default="zig-out/bin/openpuffer")
    a = p.parse_args()
    port = free_port()
    server = subprocess.Popen([a.bin, "serve", "--port", str(port), "--workers", "2"],
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    failures = []

    def check(name, ok, detail):
        print(f"{'PASS' if ok else 'FAIL'} {name:<13} {detail}", flush=True)
        if not ok:
            failures.append(name)

    try:
        deadline = time.time() + 10
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                if time.time() > deadline or server.poll() is not None:
                    raise SystemExit("server did not start")
                time.sleep(0.05)
        new = lambda: http.client.HTTPConnection("127.0.0.1", port, timeout=TIMEOUT)  # noqa: E731

        c = new()
        request(c, "POST", "/v2/namespaces/k", {"upsert_columns": {"id": [1, 2], "vector": [[1, 0], [0, 1]]}})
        c.close()
        base_fds = open_fds(server.pid)

        idle = []
        t = time.time()
        try:
            for _ in range(IDLE):
                conn = new()
                request(conn, "GET", "/v2/namespaces/k")
                idle.append(conn)  # stays open and silent
        except (socket.timeout, TimeoutError, OSError):
            check("idle-clients", False, f"connection {len(idle) + 1} got no answer within {TIMEOUT:.0f}s "
                  f"while {len(idle)} earlier ones sat idle (2 workers)")
            raise SystemExit(1)

        t = time.time()
        try:
            fresh = new()
            status, _ = request(fresh, "GET", "/v2/namespaces/k")
            fresh.close()
            check("idle-clients", status == 200, f"answered in {(time.time() - t) * 1000:.0f} ms with {IDLE} idle connections and 2 workers")
        except (socket.timeout, TimeoutError, OSError) as e:
            check("idle-clients", False, f"no answer within {TIMEOUT:.0f}s with {IDLE} idle connections and 2 workers ({e.__class__.__name__})")

        ok = 0
        for conn in idle:
            try:
                ok += request(conn, "GET", "/v2/namespaces/k")[0] == 200
            except (socket.timeout, TimeoutError, OSError):
                pass
        check("parked-reuse", ok == IDLE, f"{ok}/{IDLE} idle connections answered when they spoke again")

        s = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
        one = f"GET /v2/namespaces/k HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n\r\n".encode()
        s.sendall(one + one)
        data, deadline = b"", time.time() + TIMEOUT
        while data.count(b"HTTP/1.1 200") < 2 and time.time() < deadline:
            try:
                chunk = s.recv(65536)
            except (socket.timeout, TimeoutError):
                break
            if not chunk:
                break
            data += chunk
        s.close()
        check("pipelined", data.count(b"HTTP/1.1 200") == 2, f"{data.count(b'HTTP/1.1 200')}/2 responses to two requests in one write")

        def busy(i):
            conn = new()
            good = 0
            try:
                for j in range(20):
                    good += request(conn, "POST", "/v2/namespaces/k/query", {"rank_by": ["vector", "ANN", [1, 0]], "top_k": 2})[0] == 200
            except (socket.timeout, TimeoutError, OSError):
                pass
            finally:
                conn.close()
            return good

        t = time.time()
        with cf.ThreadPoolExecutor(32) as pool:
            good = sum(pool.map(busy, range(32)))
        check("busy-clients", good == 32 * 20, f"{good}/{32 * 20} queries answered in {time.time() - t:.1f}s with {IDLE} idle connections open")

        for conn in idle:
            conn.close()
        time.sleep(0.5)
        after = open_fds(server.pid)
        if base_fds is None or after is None:
            print("SKIP fd-release     /proc not available")
        else:
            check("fd-release", after <= base_fds + 1, f"{after} open descriptors after closing idle connections (baseline {base_fds})")
    finally:
        server.kill()
        server.wait()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("keep-alive checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
