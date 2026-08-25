#!/usr/bin/env python3
"""Stable shard-key function for multi-instance openpuffer.

Writes route to exactly one shard. Queries scatter-gather across shards
that may hold the namespace (document sharding) or hit one shard
(namespace sharding).

Hash
----
FNV-1a 64-bit (offset 0xcbf29ce484222325, prime 0x100000001b3), then
``key % n_shards``. Chosen because it is stable, dependency-free, and
matches the usual “hash the id” mental model. Not cryptographic.

Canonical key bytes
-------------------
Two modes, selected by ``shard_by``:

* ``doc`` (default, the 2M scale path)
  ``utf-8(namespace) + 0x00 + utf-8(decimal_id)``

  Integer ids are encoded as ASCII decimal with no sign, no leading
  zeros (``42`` not ``042``). ``42`` and ``"42"`` therefore hash the
  same. Empty namespace is allowed and still contributes the NUL
  separator so ``("","1")`` ≠ ``("1","")``.

* ``namespace``
  ``utf-8(namespace)`` only. Every document of a tenant lands on one
  shard. Queries can skip scatter-gather. A single 2M-doc namespace
  still has to fit in that one process — this mode does **not** solve
  the 16 GiB address-space problem.

Do not change the encoding without bumping a version: existing
deployments would reshuffle every document.
"""

from __future__ import annotations

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x00000100000001B3
SHARD_BY_DOC = "doc"
SHARD_BY_NAMESPACE = "namespace"


def fnv1a64(data: bytes) -> int:
    h = FNV_OFFSET
    for b in data:
        h ^= b
        h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def canonical_id(doc_id) -> str:
    """ASCII decimal for ints; strip for strings so ' 42 ' == 42."""
    if isinstance(doc_id, bool):
        raise TypeError("bool is not a document id")
    if isinstance(doc_id, int):
        if doc_id < 0:
            raise ValueError("document id must be non-negative")
        return str(doc_id)
    if isinstance(doc_id, str):
        s = doc_id.strip()
        if s.isdigit():
            return str(int(s))
        return s
    raise TypeError(f"unsupported document id type: {type(doc_id)!r}")


def shard_key_bytes(namespace: str, doc_id, shard_by: str = SHARD_BY_DOC) -> bytes:
    if shard_by == SHARD_BY_NAMESPACE:
        return namespace.encode("utf-8")
    if shard_by != SHARD_BY_DOC:
        raise ValueError(f"unknown shard_by {shard_by!r}")
    return namespace.encode("utf-8") + b"\x00" + canonical_id(doc_id).encode("utf-8")


def shard_for(namespace: str, doc_id, n_shards: int, shard_by: str = SHARD_BY_DOC) -> int:
    """Return owning shard index in ``[0, n_shards)``."""
    if n_shards <= 0:
        raise ValueError("n_shards must be > 0")
    return fnv1a64(shard_key_bytes(namespace, doc_id, shard_by)) % n_shards


def selftest() -> None:
    assert fnv1a64(b"") == FNV_OFFSET
    assert shard_for("ns", 42, 4) == shard_for("ns", "42", 4)
    assert shard_for("ns", 42, 4) == shard_for("ns", "042", 4)
    assert shard_for("a", 1, 8) != shard_for("b", 1, 8)
    # namespace mode ignores the document id
    assert shard_for("tenant", 1, 4, SHARD_BY_NAMESPACE) == shard_for(
        "tenant", 999, 4, SHARD_BY_NAMESPACE
    )
    # empty ns still separated from a ns that equals the id
    assert shard_key_bytes("", 1) != shard_key_bytes("1", "")
    # deterministic fixture (do not change: this IS the on-wire contract)
    assert shard_for("serve-bench", 0, 4) == 3
    assert shard_for("serve-bench", 1, 4) == 0
    assert shard_for("serve-bench", 2, 4) == 1
    assert shard_for("serve-bench", 3, 4) == 2
    seen = {shard_for("serve-bench", i, 4) for i in range(64)}
    assert seen == {0, 1, 2, 3}


if __name__ == "__main__":
    selftest()
    print("shard_key selftest ok")
