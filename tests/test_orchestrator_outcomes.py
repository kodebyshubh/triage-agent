"""Trace shape -> final_action mapping, with the agent loop and every agent mocked.

The fake agent calls the real tool closures in a scripted order, so this checks
the orchestrator's bookkeeping, not model judgment.
"""

import asyncio

import pytest
from langchain_core.tools import Tool

from app.agents import orchestrator
from app.schemas import (
    ClassificationResult,
    EscalationSummary,
    ResolutionDraft,
    RetrievalResult,
    TicketInput,
)

_TICKET = TicketInput(ticket_id="t1", user_message="help with acc-1001", session_id="s1")


@pytest.fixture
def mocked_agents(monkeypatch):
    monkeypatch.setattr(
        orchestrator,
        "classify",
        lambda t: ClassificationResult(category="billing", urgency=2, sentiment="neutral"),
    )
    monkeypatch.setattr(
        orchestrator,
        "retrieve",
        lambda q: RetrievalResult(chunks=["c"], sources=["billing.md"], confidence=0.7),
    )
    monkeypatch.setattr(
        orchestrator,
        "draft_response",
        lambda t, r: ResolutionDraft(draft_text="d", confidence=0.8, citations=["billing.md"]),
    )
    monkeypatch.setattr(
        orchestrator,
        "escalate",
        lambda t, reason: EscalationSummary(
            summary="s", suggested_priority="high", flagged_reason=reason
        ),
    )
    monkeypatch.setattr(orchestrator, "get_chat_model", lambda **_: None)


def _script(monkeypatch, steps):
    """steps: list of (tool_name, arg) the fake agent will call, in order."""

    class _Agent:
        def __init__(self, tools):
            self.tools = {t.name: t for t in tools}

        async def ainvoke(self, _payload):
            for name, arg in steps:
                await self.tools[name].ainvoke(arg)

    monkeypatch.setattr(orchestrator, "create_agent", lambda model, tools, **k: _Agent(tools))


def _mcp_tool():
    async def _lookup(account_id: str):
        return {"account_id": account_id, "plan_tier": "pro"}

    return Tool.from_function(func=None, coroutine=_lookup, name="check_account_status",
                              description="mock")


def _run(**kwargs):
    return asyncio.run(orchestrator.run_orchestrator(_TICKET, **kwargs))


def test_classify_then_draft_is_auto_resolved(monkeypatch, mocked_agents):
    _script(monkeypatch, [("classify", ""), ("retrieve_and_draft", "refund")])
    result = _run()
    assert result.final_action == "auto_resolved"
    assert [s.agent_name for s in result.trace] == ["Classifier", "Retriever+Responder"]


def test_classify_then_escalate_is_escalated(monkeypatch, mocked_agents):
    _script(monkeypatch, [("classify", ""), ("escalate", "urgent")])
    result = _run()
    assert result.final_action == "escalated"
    assert result.response.flagged_reason == "urgent"


def test_lookup_then_terminal_tool_counts_normally(monkeypatch, mocked_agents):
    _script(
        monkeypatch,
        [("classify", ""), ("check_account_status", "acc-1001"), ("retrieve_and_draft", "q")],
    )
    result = _run(mcp_tools=[_mcp_tool()])
    assert result.final_action == "auto_resolved"
    assert result.trace[1].agent_name == "AccountLookup"


def test_trace_ending_on_lookup_is_incomplete_lookup(monkeypatch, mocked_agents):
    _script(monkeypatch, [("classify", ""), ("check_account_status", "acc-1001")])
    result = _run(mcp_tools=[_mcp_tool()])
    assert result.final_action == "escalated_incomplete_lookup"


def test_empty_trace_raises(monkeypatch, mocked_agents):
    _script(monkeypatch, [])
    with pytest.raises(RuntimeError, match="never called a tool"):
        _run()


def test_skipping_classify_raises(monkeypatch, mocked_agents):
    _script(monkeypatch, [("retrieve_and_draft", "q")])
    with pytest.raises(RuntimeError, match="skipped classification"):
        _run()


def test_history_is_prepended_as_context(monkeypatch, mocked_agents):
    seen = {}

    class _Agent:
        def __init__(self, tools):
            self.tools = {t.name: t for t in tools}

        async def ainvoke(self, payload):
            seen["messages"] = payload["messages"]
            await self.tools["classify"].ainvoke("")
            await self.tools["escalate"].ainvoke("r")

    monkeypatch.setattr(orchestrator, "create_agent", lambda model, tools, **k: _Agent(tools))
    prior = _run()
    asyncio.run(orchestrator.run_orchestrator(_TICKET, history=[(_TICKET, prior)]))
    assert len(seen["messages"]) == 2
    assert seen["messages"][0][1].startswith("Context from earlier in this ticket thread")
