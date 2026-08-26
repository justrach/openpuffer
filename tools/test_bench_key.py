#!/usr/bin/env python3
"""Fixtures for hardware-keyed baselines (#18)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

from bench_key import (  # noqa: E402
    ENGINE_PROFILE,
    infer_engine_key,
    keys_compatible,
    make_key,
)
from loop_once import best_engine, decide, ledger_rows  # noqa: E402

FIXTURE = """
| id | date | hypothesis | change | files | gate | p50 (ms) | recall@10 | delta | verdict | track | notes |
| E009 | 2026-08-23 | prefetch | one-ahead | src/hnsw.zig | pass | 0.830 | 0.3130 | -11% | keep | engine | M1 Mac |
| E022 | 2026-08-23 | visited prefetch | guard | src/hnsw.zig | pass | 0.710 / 0.575 | 0.3130 | -6.7% | keep | engine | Linux Xeon AVX-512 |
| E025 | 2026-08-25 | flatten | slabs | src/hnsw.zig | pass | 0.744 / 0.581 | 0.3130 | +1% | keep | memory | MEMORY keep |
"""


def test_matching_host_only() -> None:
    rows = ledger_rows(FIXTURE)
    m1 = make_key(ENGINE_PROFILE, system="Darwin", machine="arm64", brand="Apple M1")
    m4 = make_key(ENGINE_PROFILE, system="Darwin", machine="arm64", brand="Apple M4 Pro")
    xeon = make_key(
        ENGINE_PROFILE,
        system="Linux",
        machine="x86_64",
        cpuinfo="model name: Intel(R) Xeon(R) Processor",
    )
    assert infer_engine_key("E009", "M1 Mac")
    assert keys_compatible(m1, infer_engine_key("E009", "M1 Mac") or "")
    b_m1 = best_engine(rows, m1)
    b_xeon = best_engine(rows, xeon)
    b_m4 = best_engine(rows, m4)
    assert b_m1 == ("E009", 0.830), b_m1
    assert b_xeon == ("E022", 0.575), b_xeon
    assert b_m4 is None, b_m4
    assert "baseline" in decide(0.829, 0.3105, b_m4, m4)
    assert "key=" in decide(0.829, 0.3105, b_m4, m4)
    assert "E022" in decide(0.550, 0.3105, b_xeon, xeon)
    print("test_bench_key: matching host/profile only — ok")


if __name__ == "__main__":
    test_matching_host_only()
    from bench_key import self_test

    self_test()
    print("all key fixtures passed")
