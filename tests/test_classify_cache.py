import pytest

from app.agents import classify_cache
from app.llm import TransportFailure
from app.schemas import ClassificationResult, TicketInput


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    monkeypatch.setenv("CLASSIFY_CACHE_SIZE", "2")
    monkeypatch.setenv("CLASSIFY_CACHE_TTL_S", "600")
    classify_cache.clear()
    yield
    classify_cache.clear()


def _ticket(msg):
    return TicketInput(ticket_id="t", user_message=msg, session_id="s")


def _counting_classifier():
    calls = []

    def _classify(ticket):
        calls.append(ticket.user_message)
        return ClassificationResult(category="billing", urgency=2, sentiment="neutral")

    return calls, _classify


def test_identical_message_hits_cache():
    calls, fn = _counting_classifier()
    cached = classify_cache.cached(fn, lambda: "m")
    first, second = cached(_ticket("refund")), cached(_ticket("refund"))
    assert calls == ["refund"]
    assert first == second and first is not second  # copies, not a shared object


def test_different_model_is_a_different_key():
    calls, fn = _counting_classifier()
    model = {"name": "a"}
    cached = classify_cache.cached(fn, lambda: model["name"])
    cached(_ticket("refund"))
    model["name"] = "b"
    cached(_ticket("refund"))
    assert len(calls) == 2


def test_lru_eviction():
    calls, fn = _counting_classifier()
    cached = classify_cache.cached(fn, lambda: "m")
    for msg in ("a", "b", "c", "a"):  # size 2: "a" evicted by "c"
        cached(_ticket(msg))
    assert calls == ["a", "b", "c", "a"]


def test_ttl_expiry(monkeypatch):
    monkeypatch.setenv("CLASSIFY_CACHE_TTL_S", "0")
    calls, fn = _counting_classifier()
    cached = classify_cache.cached(fn, lambda: "m")
    cached(_ticket("a"))
    cached(_ticket("a"))
    assert len(calls) == 2


def test_disabled_with_size_zero(monkeypatch):
    monkeypatch.setenv("CLASSIFY_CACHE_SIZE", "0")
    calls, fn = _counting_classifier()
    cached = classify_cache.cached(fn, lambda: "m")
    cached(_ticket("a"))
    cached(_ticket("a"))
    assert len(calls) == 2


def test_failures_are_not_cached():
    attempts = []

    def _flaky(ticket):
        attempts.append(1)
        if len(attempts) == 1:
            raise TransportFailure("malformed")
        return ClassificationResult(category="billing", urgency=2, sentiment="neutral")

    cached = classify_cache.cached(_flaky, lambda: "m")
    with pytest.raises(TransportFailure):
        cached(_ticket("a"))
    assert cached(_ticket("a")).urgency == 2
    assert len(attempts) == 2
