"""Confidence = 1 - cosine distance of the top hit, clamped to [0, 1]. ChromaDB is mocked."""

import pytest

from app import retrieval


class _FakeCollection:
    def __init__(self, distances):
        self.distances = distances

    def query(self, query_texts, n_results):
        n = len(self.distances)
        return {
            "documents": [[f"chunk {i}" for i in range(n)]],
            "metadatas": [[{"source": f"doc{i}.md"} for i in range(n)]],
            "distances": [self.distances],
        }


class _FakeClient:
    def __init__(self, distances):
        self.collection = _FakeCollection(distances)

    def get_collection(self, _name):
        return self.collection


@pytest.mark.parametrize(
    "distances,expected",
    [
        ([0.38, 0.5, 0.7], 0.62),  # uses the top hit only
        ([0.0], 1.0),
        ([1.4], 0.0),  # cosine distance can exceed 1; clamped at 0
        ([-0.05], 1.0),  # float noise below 0; clamped at 1
    ],
)
def test_confidence_math(monkeypatch, distances, expected):
    monkeypatch.setattr(retrieval, "_client", _FakeClient(distances))
    result = retrieval.retrieve("refund")
    assert result.confidence == pytest.approx(expected)
    assert result.sources == [f"doc{i}.md" for i in range(len(distances))]


def test_no_results_gives_zero_confidence(monkeypatch):
    monkeypatch.setattr(retrieval, "_client", _FakeClient([]))
    result = retrieval.retrieve("anything")
    assert result.confidence == 0.0
    assert result.chunks == []
