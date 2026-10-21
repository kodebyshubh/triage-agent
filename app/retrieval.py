"""RAG read path. The only module in the app that talks to ChromaDB.

Three retrieval modes, selected by RETRIEVAL_MODE (default "vector"):
- "vector": ChromaDB cosine similarity only (the original behavior)
- "hybrid": BM25 keyword ranking fused with the vector ranking by reciprocal
  rank fusion (RRF)
- "hybrid_rerank": hybrid, then the fused top candidates are re-scored by a
  small cross-encoder (ms-marco-MiniLM-L6-v2, run through onnxruntime)

In every mode, confidence keeps its original meaning: 1 - cosine distance of the
chunk that ends up ranked first, clamped to [0, 1].

The default stays "vector" because it measured best on this KB: see
eval/retrieval_eval_results.txt (hit@1 86.1% vs 73.6% hybrid and 80.6% hybrid
plus re-rank on 72 labeled queries). The other two modes are kept as tested,
opt-in alternatives.
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import chromadb

from app.schemas import RetrievalResult

CHROMA_DIR = Path(__file__).resolve().parent.parent / "chroma_db"
COLLECTION_NAME = "kb_docs"
MODES = ("vector", "hybrid", "hybrid_rerank")
RRF_K = 60  # standard RRF constant from Cormack et al. (2009)
RERANK_CANDIDATES = 10

_client = chromadb.PersistentClient(path=str(CHROMA_DIR))


@dataclass
class Hit:
    id: str
    document: str
    source: str
    distance: float


_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its me my "
    "of on or so that the this to was what when where which why will with you your".split()
)


def _tokenize(text: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]


def _vector_hits(query: str) -> list[Hit]:
    """All chunks ordered by cosine distance. The KB is small (about 60 chunks),
    so scoring every chunk is cheap and gives a distance for any chunk BM25 or
    the re-ranker might promote."""
    collection = _client.get_collection(COLLECTION_NAME)
    n = collection.count()
    if n == 0:
        return []
    results = collection.query(query_texts=[query], n_results=n)
    return [
        Hit(id=i, document=d, source=m["source"], distance=dist)
        for i, d, m, dist in zip(
            results["ids"][0], results["documents"][0], results["metadatas"][0],
            results["distances"][0],
        )
    ]


@lru_cache(maxsize=1)
def _bm25_index(_collection_version: int):
    from rank_bm25 import BM25Okapi

    data = _client.get_collection(COLLECTION_NAME).get(include=["documents"])
    ids, docs = data["ids"], data["documents"]
    return ids, BM25Okapi([_tokenize(d) for d in docs])


def _bm25_ranking(query: str) -> list[str]:
    collection = _client.get_collection(COLLECTION_NAME)
    ids, bm25 = _bm25_index(collection.count())
    scores = bm25.get_scores(_tokenize(query))
    order = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in order if scores[i] > 0]


def _rrf(rankings: list[list[str]]) -> list[str]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (RRF_K + rank)
    return sorted(fused, key=fused.get, reverse=True)


def rank(query: str, mode: str | None = None) -> list[Hit]:
    """Full ranked list of chunks for `query` under the given mode."""
    mode = mode or os.getenv("RETRIEVAL_MODE", "vector")
    if mode not in MODES:
        raise ValueError(f"Unknown RETRIEVAL_MODE {mode!r}, expected one of {MODES}")

    vector = _vector_hits(query)
    if mode == "vector" or not vector:
        return vector

    by_id = {h.id: h for h in vector}
    fused = [by_id[i] for i in _rrf([[h.id for h in vector], _bm25_ranking(query)])]
    if mode == "hybrid":
        return fused

    from app.reranker import rerank

    head, tail = fused[:RERANK_CANDIDATES], fused[RERANK_CANDIDATES:]
    scores = rerank(query, [h.document for h in head])
    head = [h for _, h in sorted(zip(scores, head), key=lambda p: p[0], reverse=True)]
    return head + tail


def retrieve(query: str, k: int = 3, mode: str | None = None) -> RetrievalResult:
    hits = rank(query, mode)[:k]
    confidence = max(0.0, min(1.0, 1 - hits[0].distance)) if hits else 0.0
    return RetrievalResult(
        chunks=[h.document for h in hits],
        sources=[h.source for h in hits],
        confidence=confidence,
    )
