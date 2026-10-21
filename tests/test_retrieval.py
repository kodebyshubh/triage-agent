"""Retrieval with ChromaDB and the cross-encoder mocked.

Confidence = 1 - cosine distance of the chunk ranked first, clamped to [0, 1],
in every retrieval mode.
"""

import pytest

from app import reranker, retrieval


class _FakeCollection:
    def __init__(self, docs: dict[str, str], distances: dict[str, float]):
        self.docs = docs
        self.distances = distances

    def count(self):
        return len(self.docs)

    def query(self, query_texts, n_results):
        order = sorted(self.docs, key=lambda i: self.distances[i])[:n_results]
        return {
            "ids": [order],
            "documents": [[self.docs[i] for i in order]],
            "metadatas": [[{"source": f"{i.split('-')[0]}.md"} for i in order]],
            "distances": [[self.distances[i] for i in order]],
        }

    def get(self, include):
        ids = list(self.docs)
        return {"ids": ids, "documents": [self.docs[i] for i in ids]}


class _FakeClient:
    def __init__(self, collection):
        self.collection = collection

    def get_collection(self, _name):
        return self.collection


@pytest.fixture
def kb(monkeypatch):
    def _install(docs, distances):
        retrieval._bm25_index.cache_clear()
        monkeypatch.setattr(retrieval, "_client", _FakeClient(_FakeCollection(docs, distances)))

    yield _install
    retrieval._bm25_index.cache_clear()


@pytest.mark.parametrize(
    "top_distance,expected",
    [
        (0.38, 0.62),
        (0.0, 1.0),
        (1.4, 0.0),  # cosine distance can exceed 1; clamped at 0
        (-0.05, 1.0),  # float noise below 0; clamped at 1
    ],
)
def test_confidence_math_vector_mode(kb, top_distance, expected):
    kb({"a-0": "refund policy", "a-1": "other"}, {"a-0": top_distance, "a-1": top_distance + 0.5})
    result = retrieval.retrieve("refund", mode="vector")
    assert result.confidence == pytest.approx(expected)
    assert result.chunks[0] == "refund policy"
    assert result.sources == ["a.md", "a.md"]


def test_empty_collection_gives_zero_confidence(kb):
    kb({}, {})
    for mode in retrieval.MODES:
        result = retrieval.retrieve("anything", mode=mode)
        assert result.confidence == 0.0
        assert result.chunks == []


def test_unknown_mode_rejected(kb):
    kb({"a-0": "x"}, {"a-0": 0.1})
    with pytest.raises(ValueError):
        retrieval.retrieve("x", mode="magic")


def test_mode_read_from_env(kb, monkeypatch):
    kb({"a-0": "x"}, {"a-0": 0.1})
    monkeypatch.setenv("RETRIEVAL_MODE", "magic")
    with pytest.raises(ValueError):
        retrieval.retrieve("x")


def test_rrf_rewards_agreement_between_rankings():
    fused = retrieval._rrf([["a", "b", "c"], ["b", "c", "a"]])
    assert fused[0] == "b"  # ranks 2 and 1 beat ranks 1 and 3
    assert set(fused) == {"a", "b", "c"}


def test_rrf_keeps_ids_only_in_one_ranking():
    assert retrieval._rrf([["a"], ["b"]]) == ["a", "b"]


def test_tokenizer_drops_stopwords_and_punctuation():
    assert retrieval._tokenize("How do I reset my 2FA?!") == ["reset", "2fa"]


def test_hybrid_promotes_exact_keyword_match(kb):
    docs = {"t-0": "clear the app cache", "t-1": "reset two factor 2fa codes", "t-2": "sync"}
    # Vector ranks t-0 first; only t-1 matches "2fa" for BM25, so fusion lifts t-1.
    kb(docs, {"t-0": 0.30, "t-1": 0.31, "t-2": 0.9})
    assert retrieval.rank("2fa", "vector")[0].id == "t-0"
    assert retrieval.rank("2fa", "hybrid")[0].id == "t-1"


def test_rerank_reorders_and_confidence_follows_new_top_chunk(kb, monkeypatch):
    docs = {"a-0": "first", "a-1": "second", "a-2": "third"}
    kb(docs, {"a-0": 0.2, "a-1": 0.4, "a-2": 0.6})
    scores = {"first": 0.1, "second": 5.0, "third": 1.0}
    monkeypatch.setattr(reranker, "rerank", lambda q, d: [scores[x] for x in d])
    result = retrieval.retrieve("anything", mode="hybrid_rerank")
    assert result.chunks == ["second", "third", "first"]
    assert result.confidence == pytest.approx(0.6)  # 1 - distance of "second"
