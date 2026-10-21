import fakeredis
import pytest

from app import memory
from app.memory import append_turn, get_history
from app.schemas import AgentTraceStep, EscalationSummary, ResolutionDraft, TicketInput, TriageResponse


@pytest.fixture(autouse=True, params=["memory", "redis"])
def backend(request, monkeypatch):
    """Every test runs against both backends, through the same interface."""
    memory._store.clear()
    if request.param == "redis":
        fake = fakeredis.FakeRedis(decode_responses=True)
        monkeypatch.setattr(memory, "_redis", fake)
        yield fake
    else:
        monkeypatch.setattr(memory, "_redis", None)
        monkeypatch.delenv("REDIS_URL", raising=False)
        yield None
    memory._store.clear()


def _turn(msg: str, resolved: bool = False):
    ticket = TicketInput(ticket_id="t", user_message=msg, session_id="s")
    if resolved:
        body = ResolutionDraft(draft_text=msg, confidence=0.8, citations=["billing.md"])
        trace = [AgentTraceStep(agent_name="Classifier", input={}, output={"urgency": 2},
                                timestamp="t", latency_ms=12.5)]
        response = TriageResponse(final_action="auto_resolved", response=body, trace=trace)
    else:
        body = EscalationSummary(summary=msg, suggested_priority="low", flagged_reason="r")
        response = TriageResponse(final_action="escalated", response=body, trace=[])
    return ticket, response


def test_unknown_session_has_empty_history():
    assert get_history("nobody") == []


def test_turns_append_in_order():
    append_turn("s1", *_turn("first"))
    append_turn("s1", *_turn("second"))
    assert [t.user_message for t, _ in get_history("s1")] == ["first", "second"]


def test_sessions_do_not_bleed():
    append_turn("s1", *_turn("mine"))
    assert get_history("s2") == []


def test_round_trip_keeps_response_type_and_fields():
    ticket, response = _turn("refund", resolved=True)
    append_turn("s1", ticket, response)
    [(t, r)] = get_history("s1")
    assert t == ticket
    assert r == response
    assert isinstance(r.response, ResolutionDraft)


def test_redis_key_gets_ttl(backend):
    if backend is None:
        pytest.skip("in-memory backend has no TTL")
    append_turn("s1", *_turn("first"))
    ttl = backend.ttl(memory._KEY_PREFIX + "s1")
    assert 0 < ttl <= memory.SESSION_TTL_S


def test_backend_selection(backend):
    assert memory.backend() == ("redis" if backend is not None else "memory")
