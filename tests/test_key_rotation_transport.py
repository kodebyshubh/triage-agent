"""429 failover at the HTTP transport. The network is mocked."""

import asyncio
import json

import httpx
import pytest

from app import llm


@pytest.fixture(autouse=True)
def _pool(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "k0")
    for i in range(1, 11):
        monkeypatch.setenv(f"GROQ_API_KEY_{i}", "")
    monkeypatch.setenv("GROQ_API_KEY_2", "k2")
    monkeypatch.setenv("GROQ_API_KEY_3", "k3")
    llm._cooldown_until.clear()
    yield
    llm._cooldown_until.clear()


def _request(model="openai/gpt-oss-20b"):
    return httpx.Request("POST", "https://api.groq.com/x", content=json.dumps({"model": model}))


def _fake_network(monkeypatch, limited: set[str]):
    seen = []

    def _respond(request):
        key = request.headers["Authorization"].removeprefix("Bearer ")
        seen.append(key)
        if key in limited:
            return httpx.Response(429, headers={"retry-after": "30"}, content=b"{}")
        return httpx.Response(200, content=b"{}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", lambda self, req: _respond(req))

    async def _async(self, req):
        return _respond(req)

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", _async)
    return seen


def test_429_fails_over_to_next_key_and_parks_limited_key(monkeypatch):
    seen = _fake_network(monkeypatch, limited={"k0"})
    for _ in range(4):
        assert llm._RotatingTransport().handle_request(_request()).status_code == 200
    assert seen.count("k0") == 1  # parked after its first 429, never tried again
    assert ("k0", "openai/gpt-oss-20b") in llm._cooldown_until


def test_cooldown_is_per_model(monkeypatch):
    _fake_network(monkeypatch, limited={"k0"})
    llm._RotatingTransport().handle_request(_request("openai/gpt-oss-20b"))
    assert llm._live_keys("openai/gpt-oss-120b", []) == ["k0", "k2", "k3"]


def test_all_keys_limited_returns_last_429(monkeypatch):
    monkeypatch.setattr(llm, "MAX_RATE_LIMIT_WAIT_S", 0)
    seen = _fake_network(monkeypatch, limited={"k0", "k2", "k3"})
    resp = llm._RotatingTransport().handle_request(_request())
    assert resp.status_code == 429
    assert sorted(seen) == ["k0", "k2", "k3"]  # each key tried once, no loop


def test_all_keys_limited_waits_for_earliest_then_retries(monkeypatch):
    limited = {"k0", "k2", "k3"}
    seen = _fake_network(monkeypatch, limited=limited)
    sleeps = []

    def _fake_sleep(seconds):
        # Simulate the cooldown window passing and the limit refilling.
        sleeps.append(seconds)
        llm._cooldown_until.clear()
        limited.clear()

    monkeypatch.setattr(llm.time, "sleep", _fake_sleep)
    resp = llm._RotatingTransport().handle_request(_request())
    assert resp.status_code == 200
    assert len(sleeps) == 1 and 0 < sleeps[0] <= 31
    assert len(seen) == 4


def test_async_transport_fails_over(monkeypatch):
    seen = _fake_network(monkeypatch, limited={"k0", "k2"})
    for _ in range(3):
        resp = asyncio.run(llm._AsyncRotatingTransport().handle_async_request(_request()))
        assert resp.status_code == 200
    assert seen.count("k3") == 3
