"""Transport retry must catch only Groq's tool_use_failed, never other errors."""

import asyncio

import pytest
from groq import BadRequestError

from app import llm
from app.agents import orchestrator
from app.llm import TransportFailure, invoke_structured_with_retry
from app.schemas import TicketInput
from tests.helpers import OTHER_BAD_REQUEST, TOOL_USE_FAILED, bad_request


class _FakeStructuredLLM:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def invoke(self, messages):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


# --- Layer 1: invoke_structured_with_retry ---

def test_structured_retry_recovers_after_tool_use_failed():
    fake = _FakeStructuredLLM([bad_request(TOOL_USE_FAILED), "ok"])
    assert invoke_structured_with_retry(fake, []) == "ok"
    assert fake.calls == 2


def test_structured_retry_exhausts_into_transport_failure():
    fake = _FakeStructuredLLM([bad_request(TOOL_USE_FAILED)] * 3)
    with pytest.raises(TransportFailure):
        invoke_structured_with_retry(fake, [])
    assert fake.calls == llm._MAX_TRANSPORT_RETRIES + 1


def test_structured_retry_reraises_other_bad_request_immediately():
    fake = _FakeStructuredLLM([bad_request(OTHER_BAD_REQUEST), "never reached"])
    with pytest.raises(BadRequestError):
        invoke_structured_with_retry(fake, [])
    assert fake.calls == 1


def test_structured_retry_does_not_catch_unrelated_exceptions():
    fake = _FakeStructuredLLM([ValueError("boom")])
    with pytest.raises(ValueError):
        invoke_structured_with_retry(fake, [])
    assert fake.calls == 1


# --- Layer 2: the orchestrator's own loop ---

_TICKET = TicketInput(ticket_id="t1", user_message="help", session_id="s1")


def _patch_agent(monkeypatch, error):
    attempts = []

    class _Agent:
        async def ainvoke(self, _payload):
            attempts.append(1)
            raise error

    monkeypatch.setattr(orchestrator, "get_chat_model", lambda **_: None)
    monkeypatch.setattr(orchestrator, "create_agent", lambda *a, **k: _Agent())
    return attempts


def test_orchestrator_retries_tool_use_failed_then_reports_distinct_outcome(monkeypatch):
    attempts = _patch_agent(monkeypatch, bad_request(TOOL_USE_FAILED))
    result = asyncio.run(orchestrator.run_orchestrator(_TICKET))
    assert len(attempts) == orchestrator._MAX_TRANSPORT_RETRIES + 1
    assert result.final_action == "escalated_due_to_error"
    assert result.trace[-1].agent_name == "TransportError"


def test_orchestrator_reraises_other_bad_request(monkeypatch):
    attempts = _patch_agent(monkeypatch, bad_request(OTHER_BAD_REQUEST))
    with pytest.raises(BadRequestError):
        asyncio.run(orchestrator.run_orchestrator(_TICKET))
    assert len(attempts) == 1


def test_orchestrator_does_not_retry_nested_transport_failure(monkeypatch):
    attempts = _patch_agent(monkeypatch, TransportFailure("nested call exhausted"))
    result = asyncio.run(orchestrator.run_orchestrator(_TICKET))
    assert len(attempts) == 1
    assert result.final_action == "escalated_due_to_error"
