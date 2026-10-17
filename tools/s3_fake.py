#!/usr/bin/env python3
"""A small S3 stand-in for persistence tests (stdlib only).

Path-style only: PUT/GET/DELETE /{bucket}/{key}, GET /{bucket}/?list-type=2&prefix=...
Signatures are not checked. It behaves like S3 where openpuffer's persistence
depends on it:

  - ListObjectsV2 pages (default 1000 keys) with IsTruncated + NextContinuationToken
  - every request waits `latency_ms` (object stores take tens of ms to first byte)
  - HTTP/1.1 keep-alive; `connections` counts TCP connections opened

Control endpoints (not S3):
  GET  /__stats   counters, plus the current fault settings
  POST /__faults  JSON; any of:
      get_503_pct, put_503_pct, delete_503_pct   answer 503 SlowDown that % of the time
      drop_pct          close the connection without answering (any method)
      fail_prefix       "openpuffer/u3/": every GET under it answers 503
      fail_put_suffix   "snapshot.bin": every PUT of a key ending so answers 503
      list_503          true: every LIST answers 503
      page_size, latency_ms

Usage: s3_fake.py --port 9100 --dir /tmp/bucket [--latency-ms 25] [--page-size 1000]
"""
import argparse
import json
import os
import random
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from xml.sax.saxutils import escape

lock = threading.Lock()
stats = {"connections": 0, "PUT": 0, "GET": 0, "LIST": 0, "DELETE": 0, "injected_503": 0, "dropped": 0}
faults = {"get_503_pct": 0.0, "put_503_pct": 0.0, "delete_503_pct": 0.0, "drop_pct": 0.0,
          "fail_prefix": "", "fail_put_suffix": "", "list_503": False, "page_size": 1000, "latency_ms": 25.0}
rng = random.Random(7)
ROOT = ""


def roll(pct):
    with lock:
        return rng.random() * 100 < pct


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def setup(self):
        super().setup()
        with lock:
            stats["connections"] += 1

    # -- helpers -------------------------------------------------------------
    def target(self):
        u = urllib.parse.urlsplit(self.path)
        bucket, _, key = u.path.lstrip("/").partition("/")
        return urllib.parse.unquote(bucket), urllib.parse.unquote(key), urllib.parse.parse_qs(u.query)

    def file(self, bucket, key):
        base = os.path.join(ROOT, bucket)
        path = os.path.normpath(os.path.join(base, key))
        if not path.startswith(base + os.sep):
            raise ValueError("key escapes the bucket")
        return path

    def send(self, code, body=b"", ctype="application/xml"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def injected(self, kind):
        """Apply latency and faults. True when the request was already answered or dropped."""
        time.sleep(faults["latency_ms"] / 1000)
        if roll(faults["drop_pct"]):
            with lock:
                stats["dropped"] += 1
            self.close_connection = True
            self.connection.shutdown(2)
            return True
        if kind and roll(faults[kind]):
            with lock:
                stats["injected_503"] += 1
            self.send(503, b"<Error><Code>SlowDown</Code><Message>Please reduce your request rate.</Message></Error>")
            return True
        return False

    # -- control -------------------------------------------------------------
    def do_POST(self):
        if self.path != "/__faults":
            return self.send(405)
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}"
        with lock:
            faults.update(json.loads(body))
            out = json.dumps(faults).encode()
        self.send(200, out, "application/json")

    # -- S3 ------------------------------------------------------------------
    def do_PUT(self):
        data = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        if self.injected("put_503_pct"):
            return
        bucket, key, _ = self.target()
        if faults["fail_put_suffix"] and key.endswith(faults["fail_put_suffix"]):
            with lock:
                stats["injected_503"] += 1
            return self.send(503, b"<Error><Code>SlowDown</Code></Error>")
        path = self.file(bucket, key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path + ".tmp", "wb") as f:
            f.write(data)
        os.replace(path + ".tmp", path)
        with lock:
            stats["PUT"] += 1
        self.send(200)

    def do_DELETE(self):
        if self.injected("delete_503_pct"):
            return
        bucket, key, _ = self.target()
        try:
            os.remove(self.file(bucket, key))
        except FileNotFoundError:
            pass
        with lock:
            stats["DELETE"] += 1
        self.send(204)

    def do_GET(self):
        if self.path == "/__stats":
            with lock:
                return self.send(200, json.dumps({**stats, **faults}).encode(), "application/json")
        bucket, key, query = self.target()
        if key == "" and query.get("list-type") == ["2"]:
            if self.injected(None):
                return
            if faults["list_503"]:
                with lock:
                    stats["injected_503"] += 1
                return self.send(503, b"<Error><Code>SlowDown</Code></Error>")
            return self.list(bucket, query)
        if self.injected("get_503_pct"):
            return
        if faults["fail_prefix"] and key.startswith(faults["fail_prefix"]):
            with lock:
                stats["injected_503"] += 1
            return self.send(503, b"<Error><Code>SlowDown</Code></Error>")
        with lock:
            stats["GET"] += 1
        try:
            with open(self.file(bucket, key), "rb") as f:
                data = f.read()
        except FileNotFoundError:
            return self.send(404, b"<Error><Code>NoSuchKey</Code></Error>")
        self.send(200, data, "application/octet-stream")

    def list(self, bucket, query):
        prefix = query.get("prefix", [""])[0]
        after = query.get("continuation-token", [""])[0]
        base = os.path.join(ROOT, bucket)
        keys = []
        for d, _, files in os.walk(base):
            for f in files:
                if not f.endswith(".tmp"):
                    k = os.path.relpath(os.path.join(d, f), base).replace(os.sep, "/")
                    if k.startswith(prefix) and k > after:
                        keys.append(k)
        keys.sort()
        size = int(faults["page_size"])
        page, truncated = keys[:size], len(keys) > size
        parts = ['<?xml version="1.0" encoding="UTF-8"?><ListBucketResult>',
                 f"<Name>{escape(bucket)}</Name><Prefix>{escape(prefix)}</Prefix>",
                 f"<KeyCount>{len(page)}</KeyCount><MaxKeys>{size}</MaxKeys>",
                 f"<IsTruncated>{'true' if truncated else 'false'}</IsTruncated>"]
        if truncated:
            parts.append(f"<NextContinuationToken>{escape(page[-1])}</NextContinuationToken>")
        parts += [f"<Contents><Key>{escape(k)}</Key></Contents>" for k in page]
        parts.append("</ListBucketResult>")
        with lock:
            stats["LIST"] += 1
        self.send(200, "".join(parts).encode())


def main():
    global ROOT
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=9100)
    p.add_argument("--dir", required=True)
    p.add_argument("--latency-ms", type=float, default=25)
    p.add_argument("--page-size", type=int, default=1000)
    a = p.parse_args()
    ROOT = os.path.abspath(a.dir)
    os.makedirs(ROOT, exist_ok=True)
    faults["latency_ms"], faults["page_size"] = a.latency_ms, a.page_size
    ThreadingHTTPServer.daemon_threads = True
    ThreadingHTTPServer.request_queue_size = 512  # Python's default backlog of 5 drops bursts
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
