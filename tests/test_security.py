"""X-API-Key auth (401), per-key rate limit (429), input limits (422). The
pipeline is mocked; no LLM calls."""

import fakeredis
import pytest
from fastapi.testclient import TestClient

from app import main, memory, security
from app.schemas import (
    MAX_USER_MESSAGE_CHARS,
    EscalationSummary,
    GuardrailResult,
    TriageResponse,
)

_TICKET = {"ticket_id": "t1", "user_message": "refund please", "session_id": "s-sec"}


@pytest.fixture
def client(monkeypatch):
    async def _guard(text):
        return GuardrailResult(is_safe=True, redacted_message=text, flags=[])

    async def _out_ok(text):
        return True

    async def _orch(ticket, history=None, mcp_tools=None):
        return TriageResponse(
            final_action="escalated",
            response=EscalationSummary(summary="s", suggested_priority="low", flagged_reason="r"),
            trace=[],
        )

    monkeypatch.setattr(main, "run_input_guardrail", _guard)
    monkeypatch.setattr(main, "run_output_guardrail", _out_ok)
    monkeypatch.setattr(main, "run_orchestrator", _orch)
    monkeypatch.setattr(memory, "_redis", None)
    monkeypatch.setenv("TRIAGEFORGE_API_KEYS", "key-a,key-b")
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "3")
    monkeypatch.setenv("RATE_LIMIT_EXEMPT_KEYS", "key-b")
    security._windows.clear()
    main.app.state.mcp_tools = []
    yield TestClient(main.app)
    security._windows.clear()


def test_missing_key_is_401(client):
    resp = client.post("/ticket", json=_TICKET)
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Missing X-API-Key header."


def test_wrong_key_is_401(client):
    resp = client.post("/ticket", json=_TICKET, headers={"X-API-Key": "nope"})
    assert resp.status_code == 401
    assert resp.json()["detail"] == "Invalid API key."


def test_valid_key_passes(client):
    assert client.post("/ticket", json=_TICKET, headers={"X-API-Key": "key-a"}).status_code == 200


def test_health_and_metrics_need_no_key(client):
    assert client.get("/health").status_code == 200
    assert client.get("/metrics").status_code == 200


def test_rate_limit_per_key_returns_429_with_retry_after(client):
    for _ in range(3):
        assert client.post("/ticket", json=_TICKET, headers={"X-API-Key": "key-a"}).status_code == 200
    resp = client.post("/ticket", json=_TICKET, headers={"X-API-Key": "key-a"})
    assert resp.status_code == 429
    assert int(resp.headers["Retry-After"]) >= 1
    assert "3 requests per minute" in resp.json()["detail"]


def test_exempt_key_is_not_rate_limited(client):
    for _ in range(5):
        assert client.post("/ticket", json=_TICKET, headers={"X-API-Key": "key-b"}).status_code == 200


def test_auth_off_rate_limits_by_ip(client, monkeypatch):
    monkeypatch.setenv("TRIAGEFORGE_API_KEYS", "")
    codes = [client.post("/ticket", json=_TICKET).status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_redis_rate_limiter(monkeypatch):
    fake = fakeredis.FakeRedis(decode_responses=True)
    assert [security._hit_redis(fake, "key:x", 2) for _ in range(3)][:2] == [None, None]
    assert security._hit_redis(fake, "key:x", 2) >= 1


def test_overlong_message_is_422_before_any_llm_call(client, monkeypatch):
    called = []

    async def _guard(text):
        called.append(text)

    monkeypatch.setattr(main, "run_input_guardrail", _guard)
    body = dict(_TICKET, user_message="x" * (MAX_USER_MESSAGE_CHARS + 1))
    resp = client.post("/ticket", json=body, headers={"X-API-Key": "key-a"})
    assert resp.status_code == 422
    assert resp.json()["detail"][0]["loc"] == ["body", "user_message"]
    assert called == []


def test_message_at_limit_is_accepted(client):
    body = dict(_TICKET, user_message="x" * MAX_USER_MESSAGE_CHARS)
    assert client.post("/ticket", json=body, headers={"X-API-Key": "key-a"}).status_code == 200
