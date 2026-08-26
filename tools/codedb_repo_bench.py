#!/usr/bin/env python3
"""Real-repository 512D retrieval gate for OpenPuffer.

The corpus cache is an external ``.npz`` with ``paths`` and normalized
``vectors`` arrays. It is intentionally not committed: repository text is
embedded by the GPU service only after the caller's secret filter has run.

Queries are recent non-merge commit subjects from a local git checkout. Files
changed by each commit provide weak semantic relevance labels; exact cosine
top-k over the same embeddings provides the authoritative ANN recall labels.
This keeps two questions separate:

* ANN quality: did OpenPuffer return the exact dense neighbors?
* embedding quality: did those neighbors include the files the change touched?

The script never reads ``.env`` files and has no CPU embedding fallback.
Inference is requested from an OpenAI-compatible GPU endpoint only when the
query-vector cache is absent or stale.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np


MODEL = "Qwen/Qwen3-Embedding-0.6B"
DIMENSIONS = 512
TASK = (
    "Given a software engineering issue, retrieve the code files that must "
    "be edited to resolve the issue"
)
RECALL_FLOOR = 0.99
SECRET_SUFFIXES = (".env", ".pem", ".key", ".p12", ".pfx", ".jks")
SECRET_NAMES = {
    ".env",
    ".dev.vars",
    ".git-credentials",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "credentials.json",
    "credentials",
    "id_dsa",
    "id_ecdsa",
    "id_ecdsa_sk",
    "id_ed25519",
    "id_ed25519_sk",
    "id_rsa",
    "secrets.json",
    "secrets.yaml",
    "secrets.yml",
    "service-account.json",
}
SECRET_DIRECTORIES = (".ssh/", ".gnupg/", ".aws/")


def normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, np.float32(1e-12))


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def batches(values: Sequence[str], size: int) -> Iterable[Sequence[str]]:
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def is_sensitive(path: str) -> bool:
    lowered = path.lower()
    name = pathlib.PurePosixPath(lowered).name
    env_name = name.startswith(".env") and (
        len(name) == 4 or name[4] in (".", "-", "_")
    )
    return (
        name in SECRET_NAMES
        or env_name
        or lowered.endswith(SECRET_SUFFIXES)
        or "/.git/" in "/" + lowered
        or any(directory in lowered for directory in SECRET_DIRECTORIES)
    )


def tracked_file_cards(repo: pathlib.Path) -> tuple[list[str], list[str]]:
    root = repo.resolve()
    completed = subprocess.run(
        ["git", "-C", os.fspath(root), "ls-files", "-z"],
        check=True,
        stdout=subprocess.PIPE,
    )
    paths: list[str] = []
    cards: list[str] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        path = raw.decode("utf-8", "replace")
        if is_sensitive(path):
            continue
        absolute = root / path
        try:
            if absolute.is_symlink() or not absolute.resolve().is_relative_to(root):
                continue
            content = absolute.read_bytes()
        except OSError:
            continue
        if not content or len(content) > 131_072 or b"\0" in content:
            continue
        text = content.decode("utf-8", "replace")
        body = text if len(text) <= 2200 else text[:1100] + "\n...\n" + text[-1100:]
        card = f"repository: {root.name}\npath: {path}\ncontent:\n{body}\npath: {path}"
        paths.append(path)
        cards.append(card)
    if not cards:
        raise RuntimeError("repository produced no safe tracked-file cards")
    return paths, cards


def post_embeddings(
    gateway: str,
    texts: Sequence[str],
    model: str,
    dimensions: int,
) -> np.ndarray:
    payload = json.dumps({
        "model": model,
        "input": list(texts),
        "dimensions": dimensions,
    }).encode("utf-8")
    last_error: Exception | None = None
    for attempt in range(7):
        request = urllib.request.Request(
            gateway.rstrip("/") + "/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                result = json.load(response)
            rows = sorted(result["data"], key=lambda row: int(row["index"]))
            vectors = np.asarray([row["embedding"] for row in rows], dtype=np.float32)
            if vectors.shape != (len(texts), dimensions):
                raise RuntimeError(f"unexpected embedding shape: {vectors.shape}")
            return normalize(vectors)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError) as error:
            last_error = error
            retryable = not isinstance(error, urllib.error.HTTPError) or error.code in (429, 503)
            if not retryable or attempt == 6:
                raise
            time.sleep(min(8.0, 0.25 * (2**attempt)))
    raise RuntimeError(f"GPU embedding request failed: {last_error}")


def load_or_embed_corpus(
    repo: pathlib.Path,
    cache_path: pathlib.Path,
    gateway: str,
    batch_size: int,
    refresh: bool,
) -> tuple[list[str], np.ndarray, bool, float]:
    if cache_path.exists() and not refresh:
        stored = np.load(cache_path, allow_pickle=False)
        paths = [str(path) for path in stored["paths"].tolist()]
        vectors = normalize(stored["vectors"])
        if vectors.shape != (len(paths), DIMENSIONS) or not np.isfinite(vectors).all():
            raise RuntimeError(f"invalid corpus cache shape/content: {vectors.shape}")
        return paths, vectors, True, 0.0

    paths, cards = tracked_file_cards(repo)
    started = time.perf_counter()
    chunks = []
    completed = 0
    for group in batches(cards, batch_size):
        chunks.append(post_embeddings(gateway, group, MODEL, DIMENSIONS))
        completed += len(group)
        if completed == len(cards) or completed % (batch_size * 8) == 0:
            elapsed = time.perf_counter() - started
            print(
                f"embedded corpus {completed}/{len(cards)} "
                f"({completed / elapsed:.1f} files/s)",
                flush=True,
            )
    vectors = normalize(np.concatenate(chunks, axis=0))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, paths=np.asarray(paths), vectors=vectors)
    return paths, vectors, False, time.perf_counter() - started


def git_query_records(
    repo: pathlib.Path,
    corpus_paths: set[str],
    query_limit: int,
    history_limit: int,
) -> list[dict[str, Any]]:
    completed = subprocess.run(
        [
            "git",
            "-C",
            os.fspath(repo),
            "log",
            "--no-merges",
            f"-n{history_limit}",
            "--format=@@@%H%x09%s",
            "--name-only",
        ],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    seen_queries: set[str] = set()

    def finish() -> None:
        nonlocal current
        if current is None or len(records) >= query_limit:
            current = None
            return
        gold = list(dict.fromkeys(path for path in current["changed"] if path in corpus_paths))
        subject = current["subject"].strip()
        if gold and subject and subject not in seen_queries:
            formatted = f"Instruct: {TASK}\nQuery:{subject}"
            records.append({
                "commit": current["commit"],
                "subject": subject,
                "query": formatted,
                "gold": gold,
            })
            seen_queries.add(subject)
        current = None

    for raw_line in completed.stdout.splitlines():
        if raw_line.startswith("@@@"):
            finish()
            header = raw_line[3:]
            commit, sep, subject = header.partition("\t")
            if sep:
                current = {"commit": commit, "subject": subject, "changed": []}
        elif raw_line and current is not None:
            current["changed"].append(raw_line)
    finish()
    if len(records) < query_limit:
        print(
            f"warning: requested {query_limit} queries but only {len(records)} recent "
            "commits changed files present in the corpus cache",
            file=sys.stderr,
        )
    if not records:
        raise RuntimeError("no commit queries overlap the cached corpus paths")
    return records


def load_or_embed_queries(
    records: Sequence[dict[str, Any]],
    cache_path: pathlib.Path,
    gateway: str,
    batch_size: int,
) -> tuple[np.ndarray, bool, float]:
    queries = [record["query"] for record in records]
    commits = [record["commit"] for record in records]
    if cache_path.exists():
        stored = np.load(cache_path, allow_pickle=False)
        if (
            stored["queries"].tolist() == queries
            and stored["commits"].tolist() == commits
            and stored["vectors"].shape == (len(records), DIMENSIONS)
        ):
            return normalize(stored["vectors"]), True, 0.0

    started = time.perf_counter()
    vectors = [
        post_embeddings(gateway, group, MODEL, DIMENSIONS)
        for group in batches(queries, batch_size)
    ]
    matrix = normalize(np.concatenate(vectors, axis=0))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        queries=np.asarray(queries),
        commits=np.asarray(commits),
        vectors=matrix,
    )
    return matrix, False, time.perf_counter() - started


def http_json(url: str, payload: dict[str, Any] | None = None, timeout: float = 120) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST" if data is not None else "GET",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def wait_for_server(server: subprocess.Popen[bytes], port: int) -> None:
    url = f"http://127.0.0.1:{port}/v2/namespaces/"
    for _ in range(200):
        try:
            with urllib.request.urlopen(url, timeout=1):
                pass
            return
        except urllib.error.HTTPError as error:
            # The empty namespace path deliberately returns 400, but receiving
            # an HTTP response proves the listener and request loop are ready.
            if error.code < 500:
                return
        except (urllib.error.URLError, TimeoutError):
            pass
        finally:
            if server.poll() is not None:
                output = server.stdout.read().decode("utf-8", "replace") if server.stdout else ""
                raise RuntimeError(f"openpuffer server exited early:\n{output}")
        time.sleep(0.05)
    raise RuntimeError("timed out waiting for openpuffer server")


def percentile_ms(samples: Sequence[float], fraction: float) -> float:
    ordered = sorted(samples)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index] * 1000.0


def ndcg_at_k(ranked_paths: Sequence[str], gold: set[str], k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank, path in enumerate(ranked_paths[:k])
        if path in gold
    )
    ideal = sum(1.0 / math.log2(rank + 2) for rank in range(min(len(gold), k)))
    return dcg / ideal if ideal else 0.0


def ranking_metrics(
    ranked_ids: Sequence[Sequence[int]],
    exact_ids: np.ndarray,
    paths: Sequence[str],
    records: Sequence[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    exact_hits = 0
    top1 = 0
    gold_hits = 0
    reciprocal_ranks = 0.0
    ndcgs = 0.0
    for query_index, got in enumerate(ranked_ids):
        exact = exact_ids[query_index, :k].tolist()
        exact_hits += len(set(got[:k]) & set(exact))
        top1 += bool(got and got[0] == exact[0])
        ranked_paths = [paths[index] for index in got[:k]]
        gold = set(records[query_index]["gold"])
        first = next((rank for rank, path in enumerate(ranked_paths) if path in gold), None)
        gold_hits += first is not None
        reciprocal_ranks += 0.0 if first is None else 1.0 / (first + 1)
        ndcgs += ndcg_at_k(ranked_paths, gold, k)
    count = len(records)
    return {
        "ann_recall_at_k": exact_hits / (count * k),
        "ann_top1_agreement": top1 / count,
        "gold_hit_at_k": gold_hits / count,
        "gold_mrr_at_k": reciprocal_ranks / count,
        "gold_ndcg_at_k": ndcgs / count,
    }


def query_once(
    base: str,
    vector: np.ndarray,
    k: int,
    ef: int,
    rerank_mult: int,
) -> list[int]:
    response = http_json(
        base + "/query",
        {
            "rank_by": ["vector", "ANN", vector.tolist()],
            "top_k": k,
            "ef": ef,
            "rerank_mult": rerank_mult,
        },
    )
    return [int(row["id"]) for row in response.get("rows", [])]


def sweep_configs(
    base: str,
    query_vectors: np.ndarray,
    exact_ids: np.ndarray,
    paths: Sequence[str],
    records: Sequence[dict[str, Any]],
    k: int,
    ef_values: Sequence[int],
    rerank_values: Sequence[int],
    repeats: int,
    warmup: int,
) -> list[dict[str, Any]]:
    configs = [(ef, rerank) for ef in ef_values for rerank in rerank_values]
    latencies = {config: [] for config in configs}
    rankings: dict[tuple[int, int], list[list[int] | None]] = {
        config: [None] * len(query_vectors) for config in configs
    }

    for ef, rerank in configs:
        for vector in query_vectors[:warmup]:
            query_once(base, vector, k, ef, rerank)

    # Interleave configurations per query so machine drift and config order do
    # not make the last (usually default/control) configuration look slower.
    for repeat in range(repeats):
        for query_index, vector in enumerate(query_vectors):
            order = configs.copy()
            random.Random(7 + repeat * len(query_vectors) + query_index).shuffle(order)
            for config in order:
                ef, rerank = config
                started = time.perf_counter()
                got = query_once(base, vector, k, ef, rerank)
                latencies[config].append(time.perf_counter() - started)
                if repeat == 0:
                    rankings[config][query_index] = got
                elif got != rankings[config][query_index]:
                    raise RuntimeError(
                        f"non-deterministic ranking for query={query_index} "
                        f"ef={ef}, rerank_mult={rerank}"
                    )

    results = []
    for ef, rerank in configs:
        config = (ef, rerank)
        stable_rankings = [row for row in rankings[config] if row is not None]
        if len(stable_rankings) != len(query_vectors):
            raise RuntimeError(f"incomplete rankings for ef={ef}, rerank_mult={rerank}")
        results.append({
            "ef": ef,
            "rerank_mult": rerank,
            "p50_ms": percentile_ms(latencies[config], 0.50),
            "p95_ms": percentile_ms(latencies[config], 0.95),
            "p99_ms": percentile_ms(latencies[config], 0.99),
            **ranking_metrics(stable_rankings, exact_ids, paths, records, k),
        })
    return results


def exact_semantic_metrics(
    exact_ids: np.ndarray,
    paths: Sequence[str],
    records: Sequence[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    ranked = [row[:k].tolist() for row in exact_ids]
    metrics = ranking_metrics(ranked, exact_ids, paths, records, k)
    return {
        "gold_hit_at_k": metrics["gold_hit_at_k"],
        "gold_mrr_at_k": metrics["gold_mrr_at_k"],
        "gold_ndcg_at_k": metrics["gold_ndcg_at_k"],
    }


def self_test() -> None:
    matrix = normalize(np.asarray([[3, 4], [0, 2]], dtype=np.float32))
    assert np.allclose(np.linalg.norm(matrix, axis=1), 1.0)
    assert parse_csv_ints("24, 64,128") == [24, 64, 128]
    assert is_sensitive(".env")
    assert is_sensitive("config/.env.production")
    assert is_sensitive("prod.env")
    assert is_sensitive("private_key.pem")
    assert is_sensitive("id_rsa")
    assert is_sensitive("home/.ssh/config")
    assert not is_sensitive(".envoy")
    assert not is_sensitive("src/credentials_parser.zig")
    paths = ["a.zig", "b.zig", "c.zig"]
    records = [{"gold": ["a.zig"]}, {"gold": ["c.zig"]}]
    exact = np.asarray([[0, 1], [2, 1]])
    metrics = ranking_metrics([[0, 1], [1, 2]], exact, paths, records, 2)
    assert metrics["ann_recall_at_k"] == 1.0
    assert metrics["ann_top1_agreement"] == 0.5
    assert metrics["gold_hit_at_k"] == 1.0
    assert metrics["gold_mrr_at_k"] == 0.75
    print("codedb_repo_bench self-test ok")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=pathlib.Path)
    parser.add_argument("--corpus-cache", type=pathlib.Path)
    parser.add_argument("--query-cache", type=pathlib.Path)
    parser.add_argument("--gateway", default="http://127.0.0.1:9000")
    parser.add_argument("--binary", default="./zig-out/bin/openpuffer")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--history", type=int, default=500)
    parser.add_argument("--k", type=int, default=24)
    parser.add_argument("--ef-sweep", type=parse_csv_ints, default=parse_csv_ints("24,32,48,64,96,128"))
    parser.add_argument("--rerank-mults", type=parse_csv_ints, default=parse_csv_ints("1,2,4"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--embed-batch", type=int, default=32)
    parser.add_argument("--corpus-embed-batch", type=int, default=32)
    parser.add_argument("--refresh-corpus", action="store_true")
    parser.add_argument("--upsert-batch", type=int, default=8)
    parser.add_argument("--port", type=int, default=8092)
    parser.add_argument("--namespace", default="openpuffer-codedb-real")
    parser.add_argument("--recall-floor", type=float, default=RECALL_FLOOR)
    parser.add_argument("--json-out", type=pathlib.Path)
    parser.add_argument("--require-quality", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return 0
    if args.repo is None or args.corpus_cache is None or args.query_cache is None:
        parser.error("--repo, --corpus-cache, and --query-cache are required")
    if args.queries <= 0 or args.history <= 0 or args.k <= 0:
        parser.error("--queries, --history, and --k must be positive")
    if args.repeats <= 0 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    if args.embed_batch <= 0 or args.corpus_embed_batch <= 0 or args.upsert_batch <= 0:
        parser.error("embedding and upsert batch sizes must be positive")

    paths, corpus, corpus_cache_hit, corpus_embed_seconds = load_or_embed_corpus(
        args.repo,
        args.corpus_cache,
        args.gateway,
        args.corpus_embed_batch,
        args.refresh_corpus,
    )
    if args.k > len(corpus):
        parser.error(f"--k {args.k} exceeds corpus size {len(corpus)}")

    records = git_query_records(args.repo, set(paths), args.queries, args.history)
    query_vectors, cache_hit, embed_seconds = load_or_embed_queries(
        records, args.query_cache, args.gateway, args.embed_batch
    )
    scores = query_vectors @ corpus.T
    exact_ids = np.argsort(-scores, axis=1)
    exact_metrics = exact_semantic_metrics(exact_ids, paths, records, args.k)

    server = subprocess.Popen(
        [args.binary, "serve", "--port", str(args.port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{args.port}/v2/namespaces/{args.namespace}"
    try:
        wait_for_server(server, args.port)
        load_started = time.perf_counter()
        for offset in range(0, len(corpus), args.upsert_batch):
            vectors = corpus[offset : offset + args.upsert_batch]
            http_json(
                base,
                {
                    "upsert_columns": {
                        "id": list(range(offset, offset + len(vectors))),
                        "vector": vectors.tolist(),
                    },
                    "distance_metric": "cosine_distance",
                },
            )
        load_seconds = time.perf_counter() - load_started

        results = sweep_configs(
            base,
            query_vectors,
            exact_ids,
            paths,
            records,
            args.k,
            args.ef_sweep,
            args.rerank_mults,
            args.repeats,
            args.warmup,
        )
    finally:
        server.terminate()
        try:
            server.wait(timeout=5)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    qualified = [row for row in results if row["ann_recall_at_k"] >= args.recall_floor]
    best = min(qualified, key=lambda row: row["p50_ms"]) if qualified else None
    report = {
        "profile": "codedb-real-files-qwen3-0.6b-d512",
        "repo": os.fspath(args.repo.resolve()),
        "corpus_cache": os.fspath(args.corpus_cache.resolve()),
        "model": MODEL,
        "dimensions": DIMENSIONS,
        "corpus_vectors": len(corpus),
        "queries": len(records),
        "k": args.k,
        "corpus_cache_hit": corpus_cache_hit,
        "corpus_embedding_seconds": corpus_embed_seconds,
        "query_cache_hit": cache_hit,
        "query_embedding_seconds": embed_seconds,
        "index_load_seconds": load_seconds,
        "exact_semantic": exact_metrics,
        "recall_floor": args.recall_floor,
        "results": results,
        "fastest_qualified": best,
    }

    print(
        f"codedb real repo: corpus={len(corpus)} queries={len(records)} "
        f"dim={DIMENSIONS} k={args.k} corpus_cache_hit={corpus_cache_hit} "
        f"query_cache_hit={cache_hit}"
    )
    print(
        f"exact dense semantic: hit@{args.k}={exact_metrics['gold_hit_at_k']:.4f} "
        f"MRR@{args.k}={exact_metrics['gold_mrr_at_k']:.4f} "
        f"nDCG@{args.k}={exact_metrics['gold_ndcg_at_k']:.4f}"
    )
    print("ef  rerank    p50ms    p95ms    p99ms  ANN-recall  top1  gold-hit  MRR")
    for row in results:
        print(
            f"{row['ef']:>3} {row['rerank_mult']:>7} "
            f"{row['p50_ms']:>8.3f} {row['p95_ms']:>8.3f} {row['p99_ms']:>8.3f} "
            f"{row['ann_recall_at_k']:>11.4f} {row['ann_top1_agreement']:>5.3f} "
            f"{row['gold_hit_at_k']:>9.4f} {row['gold_mrr_at_k']:>5.3f}"
        )
    if best:
        print(
            f"fastest qualified: ef={best['ef']} rerank_mult={best['rerank_mult']} "
            f"p50={best['p50_ms']:.3f}ms recall@{args.k}={best['ann_recall_at_k']:.4f}"
        )
    else:
        print(f"quality gate FAIL: no configuration reaches recall {args.recall_floor:.4f}", file=sys.stderr)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + "\n")
    if args.require_quality and best is None:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
