"""API layer: request ids, optional response fields, /metrics, JSON logs,
LangSmith toggle. The guardrails and orchestrator are mocked; the lifespan (MCP
subprocess) is not started."""

import json
import logging

import pytest
from fastapi.testclient import TestClient

from app import main, observability
from app.schemas import (
    AgentTraceStep,
    ClassificationResult,
    EscalationSummary,
    GuardrailResult,
    ResolutionDraft,
    TriageResponse,
)

_TICKET = {"ticket_id": "t1", "user_message": "refund please", "session_id": "s-obs"}


def _step(name, output, latency):
    return AgentTraceStep(agent_name=name, input={}, output=output, timestamp="t", latency_ms=latency)


@pytest.fixture
def client(monkeypatch):
    async def _guard(text):
        return GuardrailResult(is_safe=True, redacted_message=text, flags=[])

    async def _out_ok(text):
        return True

    async def _orch(ticket, history=None, mcp_tools=None, on_step=None):
        cls = ClassificationResult(category="billing", urgency=2, sentiment="neutral")
        draft = ResolutionDraft(draft_text="Go to Billing.", confidence=0.9, citations=["billing.md"])
        return TriageResponse(
            final_action="auto_resolved",
            response=draft,
            trace=[
                _step("Classifier", cls.model_dump(), 120.0),
                _step("Retriever+Responder", {"draft": draft.model_dump()}, 340.0),
            ],
        )

    monkeypatch.setattr(main, "run_input_guardrail", _guard)
    monkeypatch.setattr(main, "run_output_guardrail", _out_ok)
    monkeypatch.setattr(main, "run_orchestrator", _orch)
    main.app.state.mcp_tools = []
    return TestClient(main.app)


def test_response_keeps_existing_fields_and_adds_optional_ones(client):
    body = client.post("/ticket", json=_TICKET).json()
    assert {"final_action", "response", "trace"} <= body.keys()
    assert body["final_action"] == "auto_resolved"
    assert len(body["request_id"]) == 32
    assert body["latency_ms"] >= 0
    assert [s["latency_ms"] for s in body["trace"]] == [120.0, 340.0]


def test_request_id_echoed_from_header(client):
    resp = client.post("/ticket", json=_TICKET, headers={"X-Request-ID": "abc-123"})
    assert resp.headers["X-Request-ID"] == "abc-123"
    assert resp.json()["request_id"] == "abc-123"


def test_old_clients_can_parse_response_without_new_fields():
    legacy = {
        "final_action": "escalated",
        "response": {"summary": "s", "suggested_priority": "low", "flagged_reason": "r"},
        "trace": [{"agent_name": "Classifier", "input": {}, "output": {}, "timestamp": "t"}],
    }
    parsed = TriageResponse.model_validate(legacy)
    assert parsed.request_id is None and parsed.trace[0].latency_ms is None


def _metric(text: str, prefix: str) -> float:
    for line in text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[1])
    return 0.0


def test_metrics_endpoint_counts_outcomes_and_latency(client):
    before = client.get("/metrics").text
    client.post("/ticket", json=_TICKET)
    after = client.get("/metrics")
    assert after.headers["content-type"].startswith("text/plain")
    key = 'triage_final_action_total{final_action="auto_resolved"}'
    assert _metric(after.text, key) == _metric(before, key) + 1
    assert _metric(after.text, "triage_request_latency_seconds_count") == (
        _metric(before, "triage_request_latency_seconds_count") + 1
    )
    assert 'triage_step_latency_seconds_count{step="Classifier"}' in after.text


def test_metrics_count_guardrail_flags_and_rejections(client, monkeypatch):
    async def _reject(text):
        return GuardrailResult(is_safe=False, redacted_message=text, flags=["injection_heuristic"])

    monkeypatch.setattr(main, "run_input_guardrail", _reject)
    before = client.get("/metrics").text
    body = client.post("/ticket", json=_TICKET).json()
    after = client.get("/metrics").text
    assert body["final_action"] == "escalated" and body["trace"] == []
    key = 'triage_guardrail_flags_total{flag="injection_heuristic"}'
    assert _metric(after, key) == _metric(before, key) + 1


def test_transport_error_counter(client, monkeypatch):
    async def _orch_err(ticket, history=None, mcp_tools=None, on_step=None):
        return TriageResponse(
            final_action="escalated_due_to_error",
            response=EscalationSummary(summary="s", suggested_priority="medium", flagged_reason="r"),
            trace=[_step("TransportError", {"error": "x"}, None)],
        )

    monkeypatch.setattr(main, "run_orchestrator", _orch_err)
    before = client.get("/metrics").text
    client.post("/ticket", json=_TICKET)
    after = client.get("/metrics").text
    assert _metric(after, "triage_transport_errors_total") == (
        _metric(before, "triage_transport_errors_total") + 1
    )


def test_log_lines_are_json_with_request_id():
    record = logging.LogRecord("triageforge", logging.INFO, __file__, 1, "ticket_triaged", None, None)
    record.fields = {"final_action": "auto_resolved"}
    token = observability.request_id_var.set("rid-1")
    try:
        line = json.loads(observability.JsonFormatter().format(record))
    finally:
        observability.request_id_var.reset(token)
    assert line["request_id"] == "rid-1"
    assert line["msg"] == "ticket_triaged"
    assert line["final_action"] == "auto_resolved"


def test_langsmith_tracing_only_with_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "  ")
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert observability.configure_tracing() is False
    assert observability.os.environ["LANGSMITH_TRACING"] == "false"

    monkeypatch.setenv("LANGSMITH_API_KEY", "lsv2_test \r")
    assert observability.configure_tracing() is True
    assert observability.os.environ["LANGSMITH_TRACING"] == "true"
    assert observability.os.environ["LANGSMITH_API_KEY"] == "lsv2_test"


def test_transport_retries_reported_in_response_and_metrics(client, monkeypatch):
    from app import llm

    async def _orch_retrying(ticket, history=None, mcp_tools=None, on_step=None):
        llm.record_retry("json_validate_failed")
        return TriageResponse(
            final_action="escalated",
            response=EscalationSummary(summary="s", suggested_priority="low", flagged_reason="r"),
            trace=[],
        )

    monkeypatch.setattr(main, "run_orchestrator", _orch_retrying)
    before = client.get("/metrics").text
    body = client.post("/ticket", json=_TICKET).json()
    after = client.get("/metrics").text
    assert body["transport_retries"] == {"json_validate_failed": 1}
    key = 'triage_transport_retries_total{code="json_validate_failed"}'
    assert _metric(after, key) == _metric(before, key) + 1
    assert client.post("/ticket", json=_TICKET).json()["transport_retries"] == {"json_validate_failed": 1}


def test_upstream_rate_limit_is_503_not_500(client, monkeypatch):
    import httpx
    from groq import RateLimitError

    async def _orch_limited(ticket, history=None, mcp_tools=None, on_step=None):
        request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
        raise RateLimitError("429", response=httpx.Response(429, request=request), body=None)

    monkeypatch.setattr(main, "run_orchestrator", _orch_limited)
    resp = client.post("/ticket", json=_TICKET)
    assert resp.status_code == 503
    assert resp.headers["Retry-After"] == "60"
    assert "rate limit" in resp.json()["detail"]
