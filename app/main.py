from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.agents.orchestrator import run_orchestrator
from app.guardrails import run_input_guardrail, run_output_guardrail
from app.memory import append_turn, get_history
from app.schemas import EscalationSummary, ResolutionDraft, TicketInput, TriageResponse

app = FastAPI(title="TriageForge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ticket", response_model=TriageResponse)
async def submit_ticket(ticket: TicketInput) -> TriageResponse:
    guardrail_result = await run_input_guardrail(ticket.user_message)

    if not guardrail_result.is_safe:
        rejection = EscalationSummary(
            summary="Ticket rejected by input guardrail.",
            suggested_priority="low",
            flagged_reason=f"Flags: {', '.join(guardrail_result.flags)}",
        )
        return TriageResponse(final_action="escalated", response=rejection, trace=[])

    safe_ticket = ticket.model_copy(
        update={"user_message": guardrail_result.redacted_message}
    )
    history = get_history(ticket.session_id)

    response = run_orchestrator(safe_ticket, history=history)

    output_text = (
        response.response.draft_text
        if isinstance(response.response, ResolutionDraft)
        else response.response.summary
    )
    output_ok = await run_output_guardrail(output_text)
    if not output_ok:
        blocked = TriageResponse(
            final_action="escalated_output_blocked",
            response=EscalationSummary(
                summary="The drafted reply was blocked by the output guardrail and "
                "has been routed to a human.",
                suggested_priority="medium",
                flagged_reason="Output guardrail rejected the drafted response.",
            ),
            trace=response.trace,
        )
        append_turn(ticket.session_id, safe_ticket, blocked)
        return blocked

    append_turn(ticket.session_id, safe_ticket, response)
    return response
