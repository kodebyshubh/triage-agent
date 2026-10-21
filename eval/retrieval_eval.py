"""Offline retrieval eval: labeled query -> expected chunk id pairs, scored for
each retrieval mode. No LLM calls, no server needed (needs `python -m app.ingest`).

Reports hit@1, hit@3 and MRR (mean reciprocal rank of the expected chunk over the
full ranking) for vector only vs hybrid (BM25 + vector, RRF) vs hybrid + re-rank,
broken down by query style, plus per-query latency. Two diagnostic rows that are
not app modes are included for context: BM25 alone, and the cross-encoder
re-ranking the vector top 10 directly.

Queries were written by hand from the KB, one per chunk phrased the way a user
might ask ("paraphrase"), plus 12 short keyword-style queries. The same set was
used to pick the default mode, so there is no held-out split; treat the numbers
as a comparison between modes on this KB, not a general benchmark.

Usage:
    python -m eval.retrieval_eval [--show-misses]
"""

import argparse
import json
import statistics
import time
from pathlib import Path

from app import retrieval
from app.reranker import rerank

QUERIES_PATH = Path(__file__).parent / "retrieval_queries.json"


def _bm25_only(query: str) -> list[str]:
    return retrieval._bm25_ranking(query)


def _vector_rerank_top10(query: str) -> list[str]:
    hits = retrieval.rank(query, "vector")
    head = hits[: retrieval.RERANK_CANDIDATES]
    scores = rerank(query, [h.document for h in head])
    reordered = [h for _, h in sorted(zip(scores, head), key=lambda p: p[0], reverse=True)]
    return [h.id for h in reordered + hits[retrieval.RERANK_CANDIDATES :]]


def _mode(mode: str):
    return lambda query: [h.id for h in retrieval.rank(query, mode)]


SYSTEMS = {
    "vector": _mode("vector"),
    "hybrid": _mode("hybrid"),
    "hybrid_rerank": _mode("hybrid_rerank"),
    "(diag) bm25 only": _bm25_only,
    "(diag) vector_rerank": _vector_rerank_top10,
}


def evaluate(system, queries: list[dict]) -> dict:
    positions, latencies, misses = [], [], []
    for q in queries:
        start = time.perf_counter()
        ids = system(q["query"])
        latencies.append((time.perf_counter() - start) * 1000)
        pos = ids.index(q["expected"]) + 1 if q["expected"] in ids else None
        positions.append(pos)
        if pos != 1:
            misses.append((q["query"], q["expected"], ids[:3], pos))
    n = len(queries)
    latencies.sort()
    return {
        "hit@1": sum(p == 1 for p in positions) / n,
        "hit@3": sum(p is not None and p <= 3 for p in positions) / n,
        "mrr": sum(1 / p for p in positions if p) / n,
        "p50_ms": statistics.median(latencies),
        "p95_ms": latencies[min(n - 1, int(n * 0.95))],
        "misses": misses,
    }


def _table(title: str, queries: list[dict], with_latency: bool) -> dict:
    print(f"\n{title} (n={len(queries)})")
    header = f"  {'system':<24}{'hit@1':>8}{'hit@3':>8}{'MRR':>8}"
    print(header + (f"{'p50 ms':>9}{'p95 ms':>9}" if with_latency else ""))
    results = {}
    for name, system in SYSTEMS.items():
        r = results[name] = evaluate(system, queries)
        line = f"  {name:<24}{r['hit@1']:>8.1%}{r['hit@3']:>8.1%}{r['mrr']:>8.3f}"
        if with_latency:
            line += f"{r['p50_ms']:>9.1f}{r['p95_ms']:>9.1f}"
        print(line)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--show-misses", action="store_true")
    args = parser.parse_args()

    queries = json.loads(QUERIES_PATH.read_text())
    retrieval.rank(queries[0]["query"], "hybrid_rerank")  # warm-up: model load not timed

    print(f"Retrieval eval: {len(queries)} labeled queries over the ingested KB")
    results = _table("All queries", queries, with_latency=True)
    for style in ("paraphrase", "keyword"):
        _table(f"Style: {style}", [q for q in queries if q["style"] == style], with_latency=False)

    if args.show_misses:
        for name, r in results.items():
            print(f"\nNot ranked first, {name} ({len(r['misses'])}):")
            for query, expected, top3, pos in r["misses"]:
                print(f"  rank={pos}  expected={expected:<13} top3={top3}  {query}")


if __name__ == "__main__":
    main()
