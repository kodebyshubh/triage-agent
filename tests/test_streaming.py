"""POST /ticket/stream (SSE). The pipeline is mocked; the orchestrator mock
reports steps through on_step the way the real trace list does."""

import json

import pytest
from fastapi.testclient import TestClient

from app import main
from app.schemas import (
    AgentTraceStep,
    ClassificationResult,
    GuardrailResult,
    ResolutionDraft,
    TriageResponse,
)

_TICKET = {"ticket_id": "t1", "user_message": "refund please", "session_id": "s-stream"}


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        events.append((lines["event"], json.loads(lines["data"])))
    return events


@pytest.fixture
def client(monkeypatch):
    async def _guard(text):
        return GuardrailResult(is_safe=True, redacted_message=text, flags=[])

    async def _out_ok(text):
        return True

    async def _orch(ticket, history=None, mcp_tools=None, on_step=None):
        cls = ClassificationResult(category="billing", urgency=2, sentiment="neutral")
        draft = ResolutionDraft(draft_text="Go to Billing.", confidence=0.9, citations=["billing.md"])
        trace = [
            AgentTraceStep(agent_name="Classifier", input={}, output=cls.model_dump(), timestamp="t1"),
            AgentTraceStep(agent_name="Retriever+Responder", input={},
                           output={"draft": draft.model_dump()}, timestamp="t2"),
        ]
        for step in trace:
            if on_step:
                on_step(step)
        return TriageResponse(final_action="auto_resolved", response=draft, trace=trace)

    monkeypatch.setattr(main, "run_input_guardrail", _guard)
    monkeypatch.setattr(main, "run_output_guardrail", _out_ok)
    monkeypatch.setattr(main, "run_orchestrator", _orch)
    main.app.state.mcp_tools = []
    return TestClient(main.app)


def test_stream_emits_steps_in_order_then_final(client):
    resp = client.post("/ticket/stream", json=_TICKET, headers={"X-Request-ID": "rid-s"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = _parse_sse(resp.text)
    names = [e for e, _ in events]
    assert names == ["start", "input_guardrail", "step", "step", "output_guardrail", "final"]
    assert events[0][1] == {"request_id": "rid-s"}
    assert [d["agent_name"] for e, d in events if e == "step"] == ["Classifier", "Retriever+Responder"]
    final = events[-1][1]
    assert final["final_action"] == "auto_resolved"
    assert final["request_id"] == "rid-s"
    assert len(final["trace"]) == 2


def test_stream_rejected_ticket_has_no_steps(client, monkeypatch):
    async def _reject(text):
        return GuardrailResult(is_safe=False, redacted_message=text, flags=["injection_llm"])

    monkeypatch.setattr(main, "run_input_guardrail", _reject)
    events = _parse_sse(client.post("/ticket/stream", json=_TICKET).text)
    assert [e for e, _ in events] == ["start", "input_guardrail", "final"]
    assert events[1][1] == {"is_safe": False, "flags": ["injection_llm"]}
    assert events[-1][1]["final_action"] == "escalated"


def test_stream_reports_errors_as_event(client, monkeypatch):
    async def _boom(ticket, history=None, mcp_tools=None, on_step=None):
        raise RuntimeError("Orchestrator produced no trace steps")

    monkeypatch.setattr(main, "run_orchestrator", _boom)
    events = _parse_sse(client.post("/ticket/stream", json=_TICKET).text)
    assert events[-1] == ("error", {"detail": "RuntimeError: request failed"})


def test_plain_endpoint_unchanged_alongside_stream(client):
    body = client.post("/ticket", json=_TICKET).json()
    assert body["final_action"] == "auto_resolved"
    assert [s["agent_name"] for s in body["trace"]] == ["Classifier", "Retriever+Responder"]


def test_stream_uses_same_validation_and_auth(client, monkeypatch):
    monkeypatch.setenv("TRIAGEFORGE_API_KEYS", "k")
    assert client.post("/ticket/stream", json=_TICKET).status_code == 401
    too_long = dict(_TICKET, user_message="x" * 10_000)
    assert client.post("/ticket/stream", json=too_long, headers={"X-API-Key": "k"}).status_code == 422
