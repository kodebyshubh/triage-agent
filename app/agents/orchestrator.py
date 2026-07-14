"""The Orchestrator: a genuine LangChain tool-calling agent, not a hardcoded pipeline.

Uses langchain.agents.create_agent (LangChain 1.x's tool-calling-loop agent, the
current equivalent of the legacy AgentExecutor) so routing between classify,
retrieve_and_draft, and escalate is decided by the model's own reasoning at each
step, not by if/else logic in this file.
"""

from datetime import datetime, timezone

from groq import BadRequestError
from langchain.agents import create_agent
from langchain_core.tools import Tool

from app.agents.classifier import classify
from app.agents.escalator import escalate
from app.agents.responder import draft_response
from app.llm import TransportFailure, get_chat_model
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
# where llama-3.1-8b-instant systematically over-escalated even after prompt fixes.
# Scoped to this call only - classify/draft/escalate keep using the default model,
# which is proven reliable for structured output.
_ORCHESTRATOR_MODEL = "openai/gpt-oss-20b"

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
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_tools(ticket: TicketInput, trace: list[AgentTraceStep]) -> list[Tool]:
    def _classify_fn(_input: str = "") -> str:
        result = classify(ticket)
        trace.append(
            AgentTraceStep(
                agent_name="Classifier",
                input={"user_message": ticket.user_message},
                output=result.model_dump(),
                timestamp=_now(),
            )
        )
        return result.model_dump_json()

    def _retrieve_and_draft_fn(query: str) -> str:
        retrieval = retrieve(query)
        draft = draft_response(ticket, retrieval)
        trace.append(
            AgentTraceStep(
                agent_name="Retriever+Responder",
                input={"query": query},
                output={"retrieval": retrieval.model_dump(), "draft": draft.model_dump()},
                timestamp=_now(),
            )
        )
        return draft.model_dump_json()

    def _escalate_fn(reason: str) -> str:
        summary = escalate(ticket, reason)
        trace.append(
            AgentTraceStep(
                agent_name="Escalator",
                input={"reason": reason},
                output=summary.model_dump(),
                timestamp=_now(),
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


def run_orchestrator(
    ticket: TicketInput, history: list[HistoryTurn] | None = None
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
        tools = _build_tools(ticket, trace)
        agent = create_agent(
            get_chat_model(model=_ORCHESTRATOR_MODEL), tools=tools, system_prompt=_SYSTEM_PROMPT
        )
        try:
            agent.invoke({"messages": messages})
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

    raise RuntimeError(
        f"Orchestrator's last tool call was '{last_step.agent_name}', which isn't a "
        "terminal action (Retriever+Responder or Escalator)."
    )
