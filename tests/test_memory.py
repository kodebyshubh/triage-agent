import pytest

from app import memory
from app.memory import append_turn, get_history
from app.schemas import EscalationSummary, TicketInput, TriageResponse


@pytest.fixture(autouse=True)
def _clean_store():
    memory._store.clear()
    yield
    memory._store.clear()


def _turn(msg: str):
    ticket = TicketInput(ticket_id="t", user_message=msg, session_id="s")
    response = TriageResponse(
        final_action="escalated",
        response=EscalationSummary(summary=msg, suggested_priority="low", flagged_reason="r"),
        trace=[],
    )
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
