from app.llm import get_chat_model, invoke_structured_with_retry
from app.schemas import EscalationSummary, TicketInput

_SYSTEM_PROMPT = (
    "You are writing a human-handoff summary for a support ticket that needs escalation. "
    "Summarize the issue clearly, suggest a priority (low/medium/high), and state why "
    "it's being escalated in flagged_reason."
)


def escalate(ticket: TicketInput, reason: str) -> EscalationSummary:
    llm = get_chat_model(reasoning_effort="low")
    structured_llm = llm.with_structured_output(
        EscalationSummary, method="json_schema", strict=True
    )

    human_prompt = f"Ticket message: {ticket.user_message}\n\nEscalation reason: {reason}"

    return invoke_structured_with_retry(
        structured_llm,
        [
            ("system", _SYSTEM_PROMPT),
            ("human", human_prompt),
        ],
    )
