#!/usr/bin/env python3
"""Fixed 512D clustered code-retrieval profile (#20).

This is an ANN engine benchmark for the codedb shape (512-d, k=24, clustered).
Embedding-model / GPU time is measured separately and must not be mixed in.

Arguments are fixed. Do not pass --n/--dim/etc; the driver always runs:

    bench-synthetic --n 20000 --queries 500 --dim 512 --k 24 \\
        --ef 128 --rerank-mult 4 --clustered

Results go in experiments/codedb.md and are keyed by hardware. They never
move the generic 1536-d engine best.

    python3 tools/codedb_once.py
    python3 tools/codedb_once.py --parse-only --log /tmp/codedb.log
    python3 tools/codedb_once.py --parse-only --require-quality --log /tmp/codedb.log
    python3 tools/codedb_once.py --self-test
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
from bench_key import CODEDB_PROFILE, current_key, keys_compatible  # noqa: E402
from loop_once import find_zig, run  # noqa: E402

LOG = ROOT / "experiments" / "codedb.md"
DEFAULT_BENCH_LOG = Path("/tmp/openpuffer-codedb.log")
RECALL_FLOOR = 0.99
KEEP_BAR = 0.04
BUILD_REGRESS = 0.20
RSS_REGRESS = 0.15

METRIC = [
    "./zig-out/bin/openpuffer",
    "bench-synthetic",
    "--n",
    "20000",
    "--queries",
    "500",
    "--dim",
    "512",
    "--k",
    "24",
    "--ef",
    "128",
    "--rerank-mult",
    "4",
    "--clustered",
]

P50_RE = re.compile(r"openpuffer \(ANN\)\s+p50\s+([0-9.]+)ms", re.I)
P95_RE = re.compile(r"openpuffer \(ANN\)\s+p50\s+[0-9.]+ms\s+p95\s+([0-9.]+)ms", re.I)
P99_RE = re.compile(r"p99\s+([0-9.]+)ms", re.I)
RECALL_RE = re.compile(r"openpuffer recall@24 vs exact:\s+([0-9.]+)")
BUILD_RE = re.compile(r"index built:.*in\s+([0-9.]+)ms")
RSS_RE = re.compile(r"rss_mib phase=post-query current=(\d+)")
RSS_HUMAN = re.compile(r"post-query rss:\s+(\d+)\s+MiB")
KEY_RE = re.compile(r"\bkey=([^\s|]+)")


def parse_codedb(text: str) -> dict:
    def num(rx, required=True):
        m = rx.search(text)
        if not m:
            if required:
                sys.exit("could not parse codedb log:\n" + text[-2000:])
            return None
        return float(m.group(1))

    rss = num(RSS_RE, required=False)
    if rss is None:
        m = RSS_HUMAN.search(text)
        rss = float(m.group(1)) if m else None
    return {
        "p50": num(P50_RE),
        "p95": num(P95_RE),
        "p99": num(P99_RE),
        "recall": num(RECALL_RE),
        "build_ms": num(BUILD_RE),
        "rss_mib": rss,
    }


def ledger_best(text: str, host_key: str) -> dict | None:
    best = None
    for line in text.splitlines():
        if not line.startswith("| C"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 8:
            continue
        verdict = cells[6] if len(cells) > 6 else ""
        if verdict != "keep":
            continue
        notes = cells[-1]
        m = KEY_RE.search(notes)
        if not m or not keys_compatible(m.group(1), host_key):
            continue
        try:
            p50 = float(cells[2].split("/")[0].strip())
            rec = float(cells[3])
            build = float(cells[4])
            rss = float(cells[5])
        except ValueError:
            continue
        row = {"id": cells[0], "p50": p50, "recall": rec, "build_ms": build, "rss_mib": rss}
        if best is None or p50 < best["p50"]:
            best = row
    return best


def decide(sample: dict, best: dict | None, host_key: str, have_control: bool) -> str:
    if sample["recall"] < RECALL_FLOOR:
        return f"discard (recall {sample['recall']:.4f} < {RECALL_FLOOR}; key={host_key})"
    if best is None:
        return f"baseline (no compatible codedb keep for key={host_key})"
    reasons = []
    dp = (sample["p50"] - best["p50"]) / best["p50"]
    if dp > -KEEP_BAR:
        reasons.append(f"p50 {dp:+.1%} vs {best['id']} {best['p50']:.3f}")
    if sample["build_ms"] > best["build_ms"] * (1 + BUILD_REGRESS):
        reasons.append(f"build {sample['build_ms']:.0f}ms > +{BUILD_REGRESS:.0%} of {best['build_ms']:.0f}")
    if sample["rss_mib"] is not None and sample["rss_mib"] > best["rss_mib"] * (1 + RSS_REGRESS):
        reasons.append(f"rss {sample['rss_mib']:.0f} > +{RSS_REGRESS:.0%} of {best['rss_mib']:.0f}")
    if not reasons and dp <= -KEEP_BAR:
        if not have_control:
            return f"need control pair (p50 {sample['p50']:.3f}; key={host_key})"
        return f"keep (p50 {dp:+.1%} vs {best['id']}; key={host_key})"
    if not reasons:
        return f"neutral (within gates vs {best['id']}; key={host_key})"
    return f"discard ({'; '.join(reasons)}; key={host_key})"


def quality_gate_errors(sample: dict) -> list[str]:
    errors = []
    if sample["recall"] < RECALL_FLOOR:
        errors.append(f"recall {sample['recall']:.4f} < {RECALL_FLOOR}")
    if sample["rss_mib"] is None:
        errors.append("post-query RSS unavailable")
    return errors


def self_test() -> None:
    log = """
index built: 20000 vectors in 5800.0ms (3448 vec/s)
post-build rss: 280 MiB current, 282 MiB peak
rss_mib phase=post-build current=280 peak=282
openpuffer (ANN) p50   0.096ms  p95   0.110ms  p99   0.130ms  mean   0.098ms  10200.0 qps
openpuffer recall@24 vs exact: 1.0000
post-query rss: 284 MiB current, 286 MiB peak
rss_mib phase=post-query current=284 peak=286
"""
    s = parse_codedb(log)
    assert s["p50"] == 0.096
    assert s["recall"] == 1.0
    assert s["rss_mib"] == 284
    assert s["build_ms"] == 5800.0
    ledger = """
| id | date | p50 (ms) | recall@24 | build_ms | rss_mib | verdict | notes |
| C000 | 2026-08-26 | 0.099 | 1.0000 | 6100 | 284 | keep | key=darwin-arm64-apple-m4/codedb-20k-512-k24-ef128-rm4 |
"""
    m4 = current_key(CODEDB_PROFILE) if False else "darwin-arm64-apple-m4/codedb-20k-512-k24-ef128-rm4"
    b = ledger_best(ledger, m4)
    assert b and b["id"] == "C000"
    assert "keep" in decide({"p50": 0.090, "recall": 1.0, "build_ms": 5500, "rss_mib": 280}, b, m4, True)
    assert "baseline" in decide(s, None, "linux-x86_64-intel-xeon/" + CODEDB_PROFILE, True)
    assert quality_gate_errors(s) == []
    low_recall = dict(s)
    low_recall["recall"] = 0.7549
    assert quality_gate_errors(low_recall) == ["recall 0.7549 < 0.99"]
    print("codedb_once self-test ok")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path)
    ap.add_argument("--parse-only", action="store_true")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument(
        "--require-quality",
        action="store_true",
        help="exit nonzero when recall or RSS availability fails the codedb quality gate",
    )
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return 0

    host_key = os.environ.get("OPENPUFFER_BENCH_KEY") or current_key(CODEDB_PROFILE)
    print(f"bench key: {host_key}")
    print("profile: 512-d clustered codedb (ANN engine only; embedder time is separate)")
    best = ledger_best(LOG.read_text() if LOG.is_file() else "", host_key)
    if best:
        print(f"codedb best: {best['p50']:.3f} ms ({best['id']})")
    else:
        print("codedb best: none (need a same-host baseline)")

    if args.log:
        sample = parse_codedb(args.log.read_text())
    elif args.parse_only:
        sys.exit("need --log with --parse-only")
    else:
        zig = find_zig()
        if not args.skip_test:
            if run([zig, "build", "test"], ROOT).returncode != 0:
                print("gate FAIL")
                return 1
        if run([zig, "build", "-Doptimize=ReleaseFast"], ROOT).returncode != 0:
            print("build FAIL")
            return 1
        DEFAULT_BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_BENCH_LOG.open("w") as fh:
            proc = subprocess.run(METRIC, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print("bench FAIL; log at", DEFAULT_BENCH_LOG)
            return 1
        sample = parse_codedb(DEFAULT_BENCH_LOG.read_text())

    print(
        f"p50 {sample['p50']:.3f} ms  p95 {sample['p95']:.3f}  p99 {sample['p99']:.3f}  "
        f"recall@24 {sample['recall']:.4f}  build {sample['build_ms']:.1f} ms  "
        f"rss {sample['rss_mib']}"
    )
    have_control = os.environ.get("OPENPUFFER_CONTROL_P50") is not None
    verdict = decide(sample, best, host_key, have_control)
    print("verdict:", verdict)
    print("need ≥2 samples and a fresh compatible control before a keep")
    quality_errors = quality_gate_errors(sample)
    if args.require_quality and quality_errors:
        print("quality gate FAIL:", "; ".join(quality_errors), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
