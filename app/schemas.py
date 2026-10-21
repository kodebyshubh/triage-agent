import os
from typing import Literal

from pydantic import BaseModel, Field

MAX_USER_MESSAGE_CHARS = int(os.getenv("MAX_USER_MESSAGE_CHARS", "4000"))


class TicketInput(BaseModel):
    ticket_id: str
    user_message: str
    session_id: str


class TicketRequest(TicketInput):
    """API request body. Same fields as TicketInput, plus input limits enforced at
    the API boundary only (a 422 before any LLM call). Kept off TicketInput itself
    because PII redaction can lengthen a stored message, and reloading session
    history must never fail validation."""

    ticket_id: str = Field(max_length=200)
    user_message: str = Field(max_length=MAX_USER_MESSAGE_CHARS)
    session_id: str = Field(max_length=200)


class GuardrailResult(BaseModel):
    is_safe: bool
    redacted_message: str
    flags: list[str]


class ClassificationResult(BaseModel):
    category: Literal["billing", "technical", "account", "other"]
    urgency: int = Field(ge=1, le=5)
    sentiment: Literal["neutral", "frustrated", "urgent"]


class RetrievalResult(BaseModel):
    chunks: list[str]
    sources: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ResolutionDraft(BaseModel):
    draft_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    citations: list[str]


class EscalationSummary(BaseModel):
    summary: str
    suggested_priority: Literal["low", "medium", "high"]
    flagged_reason: str


class AgentTraceStep(BaseModel):
    agent_name: str
    input: dict
    output: dict
    timestamp: str
    # Optional, added for observability; older clients can ignore it.
    latency_ms: float | None = None


class TriageResponse(BaseModel):
    # Four distinct reasons a ticket can end up not-auto-resolved - keep them
    # separate in any eval/metrics, conflating any pair hides a real finding:
    # - "escalated": the agent genuinely judged the ticket urgent/low-confidence
    # - "escalated_due_to_error": a downstream LLM call exhausted its
    #   transport-failure retries (malformed generation), not a routing decision
    # - "escalated_output_blocked": the output guardrail rejected the drafted
    #   reply (true or false positive), not a routing decision either
    # - "escalated_incomplete_lookup": the agent called check_account_status
    #   and stopped without reaching a resolution or escalation - a model
    #   follow-through failure, not any of the above
    final_action: Literal[
        "auto_resolved",
        "escalated",
        "escalated_due_to_error",
        "escalated_output_blocked",
        "escalated_incomplete_lookup",
    ]
    response: ResolutionDraft | EscalationSummary
    trace: list[AgentTraceStep]
    # Optional observability fields, filled in by the API layer. Everything above
    # is unchanged, so existing clients keep working.
    request_id: str | None = None
    latency_ms: float | None = None
    # Provider-reported tokens per model: {model: {input_tokens, output_tokens, total_tokens}}
    token_usage: dict[str, dict[str, int]] | None = None
    # Retries after malformed model generations during this request, by Groq
    # error code. None when there were none.
    transport_retries: dict[str, int] | None = None
