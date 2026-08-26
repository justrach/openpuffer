#!/usr/bin/env python3
"""Hardware + workload keys for autoresearch baselines.

A key is `{os}-{arch}-{cpu}/{profile}`. Optional `/zig-{ver}/{mode}` suffixes
are ignored when matching legacy ledger rows.

Old engine rows without an explicit `key=` are inferred:
  E000–E011 → darwin-arm64-apple-m1 / engine-20k-1536-k10-ef128
  E012+ engine → linux-x86_64-intel-xeon / engine-20k-1536-k10-ef128

    python3 tools/bench_key.py
    python3 tools/bench_key.py --self-test
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ENGINE_PROFILE = "engine-20k-1536-k10-ef128"
CODEDB_PROFILE = "codedb-20k-512-k24-ef128-rm4"
KEY_RE = re.compile(r"\bkey=([^\s|]+)")


def _zig_version() -> str:
    env = os.environ.get("ZIG")
    zig = env if env else str(Path.home() / ".local" / "zig" / "zig")
    if not Path(zig).is_file():
        found = shutil.which("zig")
        zig = found or ""
    if not zig:
        return "unknown"
    try:
        out = subprocess.check_output([zig, "version"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.split("+")[0] if out else "unknown"


def _cpu_token(system: str, machine: str, cpuinfo: str = "", brand: str = "") -> str:
    blob = f"{cpuinfo} {brand}".lower()
    if system == "darwin":
        if "m4" in blob:
            return "apple-m4"
        if "m3" in blob:
            return "apple-m3"
        if "m2" in blob:
            return "apple-m2"
        if "m1" in blob:
            return "apple-m1"
        return "apple"
    if "epyc" in blob or ("amd" in blob and "ryzen" not in blob):
        return "amd-epyc"
    if "ryzen" in blob:
        return "amd-ryzen"
    if "xeon" in blob:
        return "intel-xeon"
    if "apple" in blob and ("m1" in blob or "m2" in blob or "m3" in blob or "m4" in blob):
        if "m4" in blob:
            return "apple-m4"
        if "m3" in blob:
            return "apple-m3"
        if "m2" in blob:
            return "apple-m2"
        return "apple-m1"
    if machine in ("arm64", "aarch64"):
        return "arm"
    return "x86"


def _arch(machine: str) -> str:
    m = machine.lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("x86_64", "amd64"):
        return "x86_64"
    return m or "unknown"


def host_parts(
    *,
    system: str | None = None,
    machine: str | None = None,
    cpuinfo: str | None = None,
    brand: str | None = None,
) -> tuple[str, str, str]:
    system = (system or platform.system()).lower()
    if system == "macos":
        system = "darwin"
    machine = machine or platform.machine()
    if cpuinfo is None and system == "linux":
        try:
            cpuinfo = Path("/proc/cpuinfo").read_text()
        except OSError:
            cpuinfo = ""
    if brand is None and system == "darwin":
        try:
            brand = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                text=True,
                timeout=5,
            ).strip()
        except (OSError, subprocess.SubprocessError):
            brand = ""
    return system, _arch(machine), _cpu_token(system, machine, cpuinfo or "", brand or "")


def make_key(
    profile: str,
    *,
    system: str | None = None,
    machine: str | None = None,
    cpuinfo: str | None = None,
    brand: str | None = None,
    zig: str | None = None,
    mode: str = "ReleaseFast",
    include_zig: bool = False,
) -> str:
    os_name, arch, cpu = host_parts(
        system=system, machine=machine, cpuinfo=cpuinfo, brand=brand
    )
    base = f"{os_name}-{arch}-{cpu}/{profile}"
    if include_zig:
        return f"{base}/zig-{zig or _zig_version()}/{mode}"
    return base


def current_key(profile: str = ENGINE_PROFILE, include_zig: bool = True) -> str:
    return make_key(profile, include_zig=include_zig)


def parse_key(key: str) -> dict[str, str]:
    parts = key.split("/")
    host = parts[0]
    profile = parts[1] if len(parts) > 1 else ""
    bits = host.split("-")
    os_name = bits[0] if bits else ""
    arch = bits[1] if len(bits) > 1 else ""
    cpu = "-".join(bits[2:]) if len(bits) > 2 else ""
    return {"os": os_name, "arch": arch, "cpu": cpu, "profile": profile, "raw": key}


def keys_compatible(a: str, b: str) -> bool:
    pa, pb = parse_key(a), parse_key(b)
    return (
        pa["os"] == pb["os"]
        and pa["arch"] == pb["arch"]
        and pa["cpu"] == pb["cpu"]
        and pa["profile"] == pb["profile"]
        and pa["os"] != ""
        and pa["profile"] != ""
    )


def infer_engine_key(eid: str, notes: str = "") -> str | None:
    if m := KEY_RE.search(notes):
        return m.group(1)
    try:
        n = int(eid[1:])
    except ValueError:
        return None
    if n <= 11:
        return f"darwin-arm64-apple-m1/{ENGINE_PROFILE}"
    return f"linux-x86_64-intel-xeon/{ENGINE_PROFILE}"


def row_key(row: dict) -> str | None:
    if row.get("track") != "engine":
        return None
    return infer_engine_key(row["id"], row.get("notes", ""))


def self_test() -> None:
    m1 = make_key(ENGINE_PROFILE, system="Darwin", machine="arm64", brand="Apple M1")
    m4 = make_key(ENGINE_PROFILE, system="Darwin", machine="arm64", brand="Apple M4 Pro")
    xeon = make_key(
        ENGINE_PROFILE,
        system="Linux",
        machine="x86_64",
        cpuinfo="model name: Intel(R) Xeon(R) Processor",
    )
    epyc = make_key(
        ENGINE_PROFILE,
        system="Linux",
        machine="x86_64",
        cpuinfo="vendor_id: AuthenticAMD\nmodel name: AMD EPYC 7763",
    )
    assert m1 == f"darwin-arm64-apple-m1/{ENGINE_PROFILE}", m1
    assert m4 == f"darwin-arm64-apple-m4/{ENGINE_PROFILE}", m4
    assert xeon == f"linux-x86_64-intel-xeon/{ENGINE_PROFILE}", xeon
    assert epyc == f"linux-x86_64-amd-epyc/{ENGINE_PROFILE}", epyc
    assert keys_compatible(m1, infer_engine_key("E009", "M1 Mac") or "")
    assert keys_compatible(xeon, infer_engine_key("E022", "") or "")
    assert not keys_compatible(m4, xeon)
    assert not keys_compatible(epyc, xeon)
    assert not keys_compatible(m1, m4)
    codedb = make_key(CODEDB_PROFILE, system="Darwin", machine="arm64", brand="Apple M4 Pro")
    assert codedb.endswith(CODEDB_PROFILE)
    assert not keys_compatible(codedb, m4)
    print("bench_key self-test ok")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    else:
        print(current_key())
