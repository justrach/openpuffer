#!/usr/bin/env python3
"""QA retrieval benchmark: openpuffer (local serve) vs turbopuffer cloud on SQuAD.

Real dataset: SQuAD v1.1 dev — Wikipedia paragraphs + real user-style questions.
Gold relevance = the paragraph the question's answer was drawn from.

Flow:
  1. download/cache SQuAD dev-v1.1
  2. take first --n unique paragraphs, first --queries questions with gold in range
  3. embed docs (RETRIEVAL_DOCUMENT) + queries (RETRIEVAL_QUERY) via Gemini
     batchEmbedContents, dim=--dim
  4. upsert into BOTH local `openpuffer serve` and turbopuffer cloud
  5. run all queries against both; measure latency, recall@k vs exact brute
     force cosine, and gold-passage hit@k

Env: GEMINI_API_KEY, TURBOPUFFER_API_KEY required.
Usage: python3 tools/qa_bench.py [--n 1000] [--queries 100] [--dim 1536]
           [--k 10] [--namespace openpuffer-squad-qa]
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
import urllib.request

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta"
TPUF_REGION = "gcp-us-central1"


def http_json(url, payload=None, headers=None, method=None, timeout=120):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("content-type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def load_squad(cache="/tmp/squad-dev-v1.1.json"):
    if not os.path.exists(cache):
        print(f"downloading SQuAD dev from {SQUAD_URL} ...")
        urllib.request.urlretrieve(SQUAD_URL, cache)
    with open(cache) as f:
        data = json.load(f)
    contexts, ctx_index = [], {}
    qa_pairs = []  # (question, gold_context_idx)
    for article in data["data"]:
        for para in article["paragraphs"]:
            text = para["context"]
            if text not in ctx_index:
                ctx_index[text] = len(contexts)
                contexts.append(text)
            for qa in para["qas"]:
                qa_pairs.append((qa["question"], ctx_index[text]))
    return contexts, qa_pairs


class Embedder:
    def __init__(self, api_key, model, dim):
        self.key = api_key
        self.model = model
        self.dim = dim

    def embed(self, texts, task_type, batch=50):
        out = []
        for i in range(0, len(texts), batch):
            chunk = texts[i : i + batch]
            reqs = [
                {
                    "model": f"models/{self.model}",
                    "taskType": task_type,
                    "outputDimensionality": self.dim,
                    "content": {"parts": [{"text": t}]},
                }
                for t in chunk
            ]
            url = f"{GEMINI_BASE}/models/{self.model}:batchEmbedContents?key={self.key}"
            resp = http_json(url, {"requests": reqs})
            out.extend(e["values"] for e in resp["embeddings"])
            print(f"  embedded {min(i + batch, len(texts))}/{len(texts)} ({task_type})")
        return out


def norm(v):
    s = sum(x * x for x in v) ** 0.5
    return [x / s for x in v]


def exact_topk(doc_vecs, q, k):
    qn = norm(q)
    scored = []
    for i, d in enumerate(doc_vecs):
        dot = sum(a * b for a, b in zip(qn, d))
        scored.append((dot, i))
    scored.sort(reverse=True)
    return [i for _, i in scored[:k]]


def percentiles(samples):
    s = sorted(samples)
    def pct(p):
        return s[min(len(s) - 1, int(p * len(s)))]
    return pct(0.50), pct(0.95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--dim", type=int, default=1536)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--namespace", default="openpuffer-squad-qa")
    ap.add_argument("--model", default="gemini-embedding-2")
    ap.add_argument("--port", type=int, default=8091)
    args = ap.parse_args()

    gemini_key = os.environ.get("GEMINI_API_KEY") or sys.exit("missing GEMINI_API_KEY")
    tpuf_key = os.environ.get("TURBOPUFFER_API_KEY") or sys.exit("missing TURBOPUFFER_API_KEY")

    # 1+2) dataset
    contexts, qa_pairs = load_squad()
    docs = contexts[: args.n]
    questions = [(q, gi) for q, gi in qa_pairs if gi < args.n][: args.queries]
    print(f"dataset: {len(docs)} paragraphs, {len(questions)} SQuAD questions")

    # 3) embeddings
    emb = Embedder(gemini_key, args.model, args.dim)
    try:
        doc_vecs = emb.embed(docs, "RETRIEVAL_DOCUMENT")
        q_vecs = emb.embed([q for q, _ in questions], "RETRIEVAL_QUERY")
    except Exception as e:
        if args.model == "gemini-embedding-2":
            print(f"model {args.model} failed ({e}); falling back to gemini-embedding-001")
            emb.model = "gemini-embedding-001"
            doc_vecs = emb.embed(docs, "RETRIEVAL_DOCUMENT")
            q_vecs = emb.embed([q for q, _ in questions], "RETRIEVAL_QUERY")
        else:
            raise

    # 4a) local openpuffer server
    binary = "./zig-out/bin/openpuffer"
    server = subprocess.Popen(
        [binary, "serve", "--port", str(args.port)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    base = f"http://127.0.0.1:{args.port}/v2/namespaces/{args.namespace}"
    for _ in range(100):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{args.port}/v2/namespaces/", timeout=1)
            break
        except Exception:
            if server.poll() is not None:
                sys.exit(f"server died:\n{server.stdout.read().decode()}")
            time.sleep(0.1)
    print(f"local server up on :{args.port}")

    B = 100
    t0 = time.time()
    for i in range(0, len(doc_vecs), B):
        chunk = doc_vecs[i : i + B]
        body = {"upsert_columns": {
            "id": list(range(i, i + len(chunk))),
            "vector": chunk,
        }, "distance_metric": "cosine_distance"}
        http_json(base, body)
    print(f"local load done in {time.time()-t0:.1f}s")

    # 4b) turbopuffer cloud
    tbase = f"https://{TPUF_REGION}.turbopuffer.com/v2/namespaces/{args.namespace}"
    auth = {"authorization": f"Bearer {tpuf_key}"}
    try:
        http_json(tbase, method="DELETE", headers=auth)
    except Exception:
        pass
    t0 = time.time()
    for i in range(0, len(doc_vecs), B):
        chunk = doc_vecs[i : i + B]
        body = {"upsert_columns": {
            "id": list(range(i, i + len(chunk))),
            "vector": chunk,
        }, "distance_metric": "cosine_distance", "encoding": None}
        http_json(tbase, body, headers=auth)
    print(f"turbopuffer load done in {time.time()-t0:.1f}s")

    # 5) queries against both
    lat_local, lat_tpuf = [], []
    ann_local_hits = ann_tpuf_hits = gold_local = gold_tpuf = 0
    total_ann = 0
    for qi, (qtext, gold_idx) in enumerate(questions):
        qv = q_vecs[qi]

        t0 = time.perf_counter()
        rloc = http_json(f"{base}/query",
                         {"rank_by": ["vector", "ANN", qv], "top_k": args.k})
        lat_local.append(time.perf_counter() - t0)
        ids_loc = [r["id"] for r in rloc.get("rows", [])]

        t0 = time.perf_counter()
        rtp = http_json(f"{tbase}/query",
                        {"rank_by": ["vector", "ANN", qv], "top_k": args.k,
                         "consistency": {"level": "strong"}},
                        headers=auth)
        lat_tpuf.append(time.perf_counter() - t0)
        ids_tp = [r["id"] for r in rtp.get("rows", [])]

        gt = exact_topk(doc_vecs, qv, args.k)
        ann_local_hits += len(set(ids_loc) & set(gt))
        ann_tpuf_hits += len(set(ids_tp) & set(gt))
        total_ann += args.k
        gold_local += gold_idx in ids_loc
        gold_tpuf += gold_idx in ids_tp

    l50, l95 = percentiles(lat_local)
    t50, t95 = percentiles(lat_tpuf)
    n = len(questions)
    print("\n=== SQuAD QA retrieval benchmark ===")
    print(f"dataset: {len(docs)} Wikipedia paragraphs (SQuAD dev), {n} real questions, "
          f"k={args.k}, dim={args.dim}, ef(local)=256")
    print(f"{'engine':<22} {'p50':>9} {'p95':>9}   recall@k(exact)  gold-hit@k(QA)")
    print(f"{'openpuffer (local)':<22} {l50*1000:8.2f}ms {l95*1000:8.2f}ms   "
          f"{ann_local_hits/total_ann:>14.4f}  {gold_local/n:>12.4f}")
    print(f"{'turbopuffer (cloud)':<22} {t50*1000:8.2f}ms {t95*1000:8.2f}ms   "
          f"{ann_tpuf_hits/total_ann:>14.4f}  {gold_tpuf/n:>12.4f}")
    print("\nnote: turbopuffer latency includes WAN round-trip; openpuffer is localhost HTTP.")
    print("note: gold-hit@k = fraction of questions where the paragraph containing the")
    print("      answer was retrieved — the metric that matters for RAG/QA.")

    server.terminate()


if __name__ == "__main__":
    main()
