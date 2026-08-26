#!/usr/bin/env python3
"""One scored-metric cycle for the openpuffer research loop.

Runs the exact program.md command, parses p50 + recall@10, and compares
against the best *engine-track* keep with a compatible hardware/workload
key (`tools/bench_key.py`). Xeon and Apple numbers are not one score.

This does not edit code or decide a hypothesis. It only makes the keep /
discard arithmetic mechanical so agents stop mixing HTTP / memory / knob
cells into the engine score.

    python3 tools/loop_once.py
    python3 tools/loop_once.py --skip-test
    python3 tools/loop_once.py --log /tmp/openpuffer-bench.log   # parse only
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "experiments" / "log.md"
sys.path.insert(0, str(ROOT / "tools"))
from bench_key import (  # noqa: E402
    ENGINE_PROFILE,
    current_key,
    keys_compatible,
    row_key,
)
DEFAULT_BENCH_LOG = Path("/tmp/openpuffer-bench.log")
METRIC = [
    "./zig-out/bin/openpuffer",
    "bench-synthetic",
    "--n",
    "20000",
    "--queries",
    "200",
    "--dim",
    "1536",
    "--k",
    "10",
    "--ef",
    "128",
]
P50_RE = re.compile(
    r"openpuffer \(ANN\)\s+p50\s+([0-9.]+)ms",
    re.IGNORECASE,
)
RECALL_RE = re.compile(
    r"openpuffer recall@10 vs exact:\s+([0-9.]+)",
    re.IGNORECASE,
)
ROW_RE = re.compile(r"^\|\s*(E\d+)\s*\|")
KEEP_BAR = 0.04
RECALL_FLOOR = 0.30


def find_zig() -> str:
    env = os.environ.get("ZIG")
    if env:
        return env
    home = Path.home() / ".local" / "zig" / "zig"
    if home.is_file():
        return str(home)
    found = shutil.which("zig")
    if found:
        return found
    sys.exit("zig not found; set ZIG= or install to ~/.local/zig/zig")


def parse_bench(text: str) -> tuple[float, float]:
    pm = P50_RE.search(text)
    rm = RECALL_RE.search(text)
    if not pm or not rm:
        sys.exit("could not parse p50/recall from bench log:\n" + text[-2000:])
    return float(pm.group(1)), float(rm.group(1))


def parse_p50_cell(cell: str) -> list[float]:
    if re.search(r"http|serial|ka\b", cell, re.IGNORECASE):
        return []
    runs = []
    for tok in cell.split("/"):
        tok = tok.strip()
        try:
            runs.append(float(tok))
        except ValueError:
            pass
    return runs


def infer_track(eid: str, p50_cell: str, notes: str) -> str:
    blob = f"{p50_cell} {notes}".lower()
    if "memory keep" in blob:
        return "memory"
    if "knob" in blob or "docs keep" in blob:
        return "knobs"
    if p50_cell.lower().startswith("http") or "serving metric" in blob:
        return "serve"
    n = int(eid[1:])
    if n <= 22:
        return "engine"
    if n in (23, 24):
        return "serve"
    if n == 25:
        return "memory"
    if n == 26:
        return "knobs"
    return "engine"


def ledger_rows(text: str) -> list[dict]:
    rows = []
    for line in text.splitlines():
        if not ROW_RE.match(line):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 11:
            continue
        eid, p50_cell, recall, verdict = cells[0], cells[6], cells[7], cells[9]
        if len(cells) >= 12:
            track, notes = cells[10], cells[11]
        else:
            notes = cells[10]
            track = infer_track(eid, p50_cell, notes)
        try:
            recall_v = float(recall) if recall not in ("—", "-", "") else None
        except ValueError:
            recall_v = None
        rows.append(
            {
                "id": eid,
                "runs": parse_p50_cell(p50_cell),
                "verdict": verdict,
                "recall": recall_v,
                "track": track,
            }
        )
    return rows


def best_engine(rows: list[dict], host_key: str) -> tuple[str, float] | None:
    kept = []
    for r in rows:
        if r["verdict"] != "keep" or r["track"] != "engine" or not r["runs"]:
            continue
        rk = row_key(r)
        if rk is None or not keys_compatible(rk, host_key):
            continue
        kept.append((r["id"], min(r["runs"])))
    if not kept:
        return None
    return min(kept, key=lambda t: t[1])


def decide(p50: float, recall: float, best: tuple[str, float] | None, host_key: str) -> str:
    if recall < RECALL_FLOOR:
        return f"discard (recall {recall:.4f} < {RECALL_FLOOR}; key={host_key})"
    if best is None:
        return f"baseline (no compatible keep for key={host_key})"
    bid, bval = best
    delta = (p50 - bval) / bval
    control = os.environ.get("OPENPUFFER_CONTROL_P50")
    if delta <= -KEEP_BAR:
        if control is None and abs(delta) < 0.08:
            return (
                f"need control pair (candidate {p50:.3f} vs {bid} {bval:.3f} "
                f"{delta:+.1%}; set OPENPUFFER_CONTROL_P50; key={host_key})"
            )
        return f"keep (≥4% vs {bid} {bval:.3f}; {delta:+.1%}; key={host_key})"
    return (
        f"discard (need ≤{-KEEP_BAR:.0%} vs {bid} {bval:.3f}; {delta:+.1%}; "
        f"key={host_key})"
    )


def run(cmd: list[str], cwd: Path) -> subprocess.CompletedProcess:
    print("+", " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=cwd, text=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, help="parse this bench log only")
    ap.add_argument("--skip-test", action="store_true")
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    rows = ledger_rows(LOG.read_text())
    host_key = os.environ.get("OPENPUFFER_BENCH_KEY") or current_key(ENGINE_PROFILE)
    best = best_engine(rows, host_key)
    print(f"bench key: {host_key}")
    if best:
        print(f"engine best: {best[1]:.3f} ms ({best[0]})")
    else:
        print("engine best: none (need a same-host baseline)")

    if args.log:
        p50, recall = parse_bench(args.log.read_text())
    else:
        zig = find_zig()
        if not args.skip_test:
            if run([zig, "build", "test"], ROOT).returncode != 0:
                print("gate FAIL")
                return 1
            print("gate pass")
        if not args.skip_build:
            if run([zig, "build", "-Doptimize=ReleaseFast"], ROOT).returncode != 0:
                print("build FAIL")
                return 1
        DEFAULT_BENCH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with DEFAULT_BENCH_LOG.open("w") as fh:
            proc = subprocess.run(METRIC, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            print("bench FAIL; log at", DEFAULT_BENCH_LOG)
            return 1
        p50, recall = parse_bench(DEFAULT_BENCH_LOG.read_text())

    print(f"p50 {p50:.3f} ms  recall@10 {recall:.4f}")
    print("verdict:", decide(p50, recall, best, host_key))
    print("log:", args.log or DEFAULT_BENCH_LOG)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
