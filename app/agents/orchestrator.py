"""The Orchestrator: a genuine LangChain tool-calling agent, not a hardcoded pipeline.

Uses langchain.agents.create_agent (LangChain 1.x's tool-calling-loop agent, the
current equivalent of the legacy AgentExecutor) so routing between classify,
retrieve_and_draft, and escalate is decided by the model's own reasoning at each
step, not by if/else logic in this file.
"""

import time
from datetime import datetime, timezone

from groq import BadRequestError
from langchain.agents import create_agent
from langchain_core.tools import BaseTool, Tool

from app.agents.classifier import classify
from app.agents.escalator import escalate
from app.agents.responder import draft_response
from app.llm import ROLE_MODELS, TransportFailure, get_chat_model
from app.retrieval import retrieve
from app.schemas import (
    AgentTraceStep,
    EscalationSummary,
    ResolutionDraft,
    TicketInput,
    TriageResponse,
)

HistoryTurn = tuple[TicketInput, TriageResponse]

# gpt-oss-20b proved reliable at the tool-calling loop itself in isolated testing,
# where the previous default (llama-3.1-8b-instant, since decommissioned by Groq)
# systematically over-escalated even after prompt fixes. The default model is now
# also gpt-oss-20b. The orchestrator keeps the model default reasoning effort
# while classify/draft/escalate run at "low". Selected per role in app/llm.py.
_ORCHESTRATOR_MODEL = ROLE_MODELS["orchestrator"]

_SYSTEM_PROMPT = """You are the Orchestrator for a support ticket triage system.

Rules you must follow exactly:
1. Call exactly ONE tool per turn. Never call two tools at the same time. After each
   tool call, wait for its result before deciding what to do next.
2. Always call `classify` first, alone, before any other tool.
3. Call `escalate` ONLY if urgency >= 4, OR if `retrieve_and_draft`'s confidence is
   below 0.4. Do not call escalate as a default or "safe" choice - if neither
   condition is met, you MUST call `retrieve_and_draft` instead.
4. If urgency is below 4, call `retrieve_and_draft` first before considering
   escalation at all - never escalate a low/medium urgency ticket without first
   checking retrieval confidence.
5. Never call `escalate` before you have seen classify's result.
6. Stop as soon as you reach a resolution or escalation - don't call more tools than
   necessary.
7. If the ticket references a specific account or account ID, call
   check_account_status before calling retrieve_and_draft or escalate.
   check_account_status is NEVER a terminal action by itself - it only gathers
   context. You MUST still call retrieve_and_draft or escalate afterward, in the
   same way you would have without it. Never stop right after
   check_account_status.
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ms_since(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)


def _build_tools(ticket: TicketInput, trace: list[AgentTraceStep]) -> list[Tool]:
    def _classify_fn(_input: str = "") -> str:
        start = time.perf_counter()
        result = classify(ticket)
        trace.append(
            AgentTraceStep(
                agent_name="Classifier",
                input={"user_message": ticket.user_message},
                output=result.model_dump(),
                timestamp=_now(),
                latency_ms=_ms_since(start),
            )
        )
        return result.model_dump_json()

    def _retrieve_and_draft_fn(query: str) -> str:
        start = time.perf_counter()
        retrieval = retrieve(query)
        draft = draft_response(ticket, retrieval)
        trace.append(
            AgentTraceStep(
                agent_name="Retriever+Responder",
                input={"query": query},
                output={"retrieval": retrieval.model_dump(), "draft": draft.model_dump()},
                timestamp=_now(),
                latency_ms=_ms_since(start),
            )
        )
        return draft.model_dump_json()

    def _escalate_fn(reason: str) -> str:
        start = time.perf_counter()
        summary = escalate(ticket, reason)
        trace.append(
            AgentTraceStep(
                agent_name="Escalator",
                input={"reason": reason},
                output=summary.model_dump(),
                timestamp=_now(),
                latency_ms=_ms_since(start),
            )
        )
        return summary.model_dump_json()

    classify_tool = Tool.from_function(
        func=_classify_fn,
        name="classify",
        description=(
            "Classify the ticket's category, urgency (1-5), and sentiment. "
            "Must be called first, exactly once, before any other tool."
        ),
    )
    retrieve_and_draft_tool = Tool.from_function(
        func=_retrieve_and_draft_fn,
        name="retrieve_and_draft",
        description=(
            "Given a short search query derived from the ticket, retrieve relevant "
            "knowledge base excerpts and draft a resolution reply with citations and "
            "a confidence score. Use for low/medium urgency tickets."
        ),
    )
    escalate_tool = Tool.from_function(
        func=_escalate_fn,
        name="escalate",
        description=(
            "Escalate the ticket to a human with a summary and suggested priority. "
            "Use ONLY when urgency is 4-5, or when retrieve_and_draft returned "
            "confidence below 0.4. Never use as a default or safe choice."
        ),
    )

    return [classify_tool, retrieve_and_draft_tool, escalate_tool]


def _build_mcp_tools(mcp_tools: list[BaseTool], trace: list[AgentTraceStep]) -> list[Tool]:
    """Wraps the check_account_status MCP tool with the same trace-logging
    pattern as the 3 in-process tools above (same AgentTraceStep fields, same
    _now() timestamp, appended to the same shared trace list). Necessarily
    async, unlike the sync tools above - MCP tool execution requires an
    active client session (see langchain_mcp_adapters), which this codebase
    doesn't otherwise use."""
    raw_tool = next((t for t in mcp_tools if t.name == "check_account_status"), None)
    if raw_tool is None:
        return []

    async def _check_account_status_fn(account_id: str) -> str:
        start = time.perf_counter()
        result = await raw_tool.ainvoke({"account_id": account_id})
        trace.append(
            AgentTraceStep(
                agent_name="AccountLookup",
                input={"account_id": account_id},
                output=result if isinstance(result, dict) else {"result": result},
                timestamp=_now(),
                latency_ms=_ms_since(start),
            )
        )
        return str(result)

    check_account_status_tool = Tool.from_function(
        func=None,
        coroutine=_check_account_status_fn,
        name="check_account_status",
        description=(
            "Look up an account's plan tier, standing, and open disputes via the "
            "MCP-connected CRM/billing tool. Call this before retrieve_and_draft or "
            "escalate if the ticket references a specific account or account ID."
        ),
    )
    return [check_account_status_tool]


def _history_to_message(history: list[HistoryTurn]) -> str | None:
    if not history:
        return None
    lines = []
    for prior_ticket, prior_response in history:
        lines.append(f"Prior message: {prior_ticket.user_message}")
        prior = prior_response.response
        if isinstance(prior, ResolutionDraft):
            lines.append(f"Prior resolution: {prior.draft_text}")
        else:
            lines.append(f"Prior escalation: {prior.summary}")
    return "Context from earlier in this ticket thread:\n" + "\n".join(lines)


_MAX_TRANSPORT_RETRIES = 2  # retries a malformed tool-call generation, never a routing decision


async def run_orchestrator(
    ticket: TicketInput,
    history: list[HistoryTurn] | None = None,
    mcp_tools: list[BaseTool] | None = None,
) -> TriageResponse:
    messages = []
    context_message = _history_to_message(history or [])
    if context_message:
        messages.append(("human", context_message))
    messages.append(("human", ticket.user_message))

    trace: list[AgentTraceStep] = []
    last_transport_error: Exception | None = None
    for _attempt in range(_MAX_TRANSPORT_RETRIES + 1):
        trace = []
        tools = _build_tools(ticket, trace) + _build_mcp_tools(mcp_tools or [], trace)
        agent = create_agent(
            get_chat_model(model=_ORCHESTRATOR_MODEL), tools=tools, system_prompt=_SYSTEM_PROMPT
        )
        try:
            await agent.ainvoke({"messages": messages})
            last_transport_error = None
            break
        except BadRequestError as e:
            # Only retry the model's own malformed function-call syntax (a transport
            # failure). A syntactically valid but disagreeable routing decision is not
            # caught here - that's a model quality issue, not something to retry away.
            if "tool_use_failed" not in str(e):
                raise
            last_transport_error = e
        except TransportFailure as e:
            # A tool call (classify/draft_response/escalate) already exhausted its
            # own retries inside invoke_structured_with_retry. Retrying the whole
            # orchestrator loop would just re-run everything and likely hit the same
            # generation glitch again, so stop here rather than looping further.
            last_transport_error = e
            break

    if last_transport_error is not None:
        error_summary = EscalationSummary(
            summary="This ticket could not be fully processed due to a repeated "
            "model transport failure and has been routed to a human.",
            suggested_priority="medium",
            flagged_reason=f"Transport failure: {last_transport_error}",
        )
        trace.append(
            AgentTraceStep(
                agent_name="TransportError",
                input={"ticket_id": ticket.ticket_id},
                output={"error": str(last_transport_error)},
                timestamp=_now(),
            )
        )
        return TriageResponse(
            final_action="escalated_due_to_error",
            response=error_summary,
            trace=trace,
        )

    if not trace:
        raise RuntimeError(
            "Orchestrator produced no trace steps - the agent never called a tool. "
            "Not adding a fallback for this; it needs investigation, not a silent patch."
        )

    if trace[0].agent_name != "Classifier":
        raise RuntimeError(
            f"Orchestrator's first tool call was '{trace[0].agent_name}', not "
            "'Classifier'. The agent skipped classification - flagging rather than "
            "forcing a hardcoded order."
        )

    last_step = trace[-1]
    if last_step.agent_name == "Escalator":
        return TriageResponse(
            final_action="escalated",
            response=EscalationSummary(**last_step.output),
            trace=trace,
        )
    if last_step.agent_name == "Retriever+Responder":
        return TriageResponse(
            final_action="auto_resolved",
            response=ResolutionDraft(**last_step.output["draft"]),
            trace=trace,
        )
    if last_step.agent_name == "AccountLookup":
        # A recognized, non-terminal follow-through failure: the model called
        # check_account_status and stopped instead of continuing to
        # retrieve_and_draft or escalate. Distinct from the generic RuntimeError
        # below, which is reserved for genuinely unrecognized trace shapes.
        return TriageResponse(
            final_action="escalated_incomplete_lookup",
            response=EscalationSummary(
                summary="The agent looked up account information but did not "
                "reach a resolution or escalation, and has been routed to a "
                "human.",
                suggested_priority="medium",
                flagged_reason="Trace ended after check_account_status without "
                "a terminal action.",
            ),
            trace=trace,
        )

    raise RuntimeError(
        f"Orchestrator's last tool call was '{last_step.agent_name}', which isn't a "
        "terminal action (Retriever+Responder or Escalator)."
    )
