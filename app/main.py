import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks import get_usage_metadata_callback
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.agents.orchestrator import run_orchestrator
from app.guardrails import run_input_guardrail, run_output_guardrail
from app.memory import append_turn, get_history
from app.observability import (
    FINAL_ACTIONS,
    GUARDRAIL_FLAGS,
    LLM_TOKENS,
    OUTPUT_GUARDRAIL_BLOCKS,
    REQUEST_ERRORS,
    REQUEST_LATENCY,
    STEP_LATENCY,
    TRANSPORT_ERRORS,
    configure_logging,
    configure_tracing,
    log_event,
    logger,
    request_id_var,
)
from app.schemas import EscalationSummary, ResolutionDraft, TicketInput, TriageResponse

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

configure_logging()
_TRACING = configure_tracing()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log_event("startup", langsmith_tracing=_TRACING)
    # MCP session held open for the app's lifetime (not reconnected per ticket)
    # so a per-request tool call doesn't pay subprocess-spawn latency on top of
    # the pipeline's already-documented sequential-LLM-call bottleneck.
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "mcp_server.server"], cwd=str(_PROJECT_ROOT)
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            app.state.mcp_tools = await load_mcp_tools(session)
            yield


app = FastAPI(title="TriageForge", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Tags every request with an id (the caller's X-Request-ID if sent) that
    appears in every log line, the response header, and the TriageResponse."""
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    token = request_id_var.set(request_id)
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as e:
        REQUEST_ERRORS.labels(error_type=type(e).__name__).inc()
        logger.exception("request_failed", extra={"fields": {"path": request.url.path}})
        raise
    finally:
        request_id_var.reset(token)
    response.headers["X-Request-ID"] = request_id
    if request.url.path != "/metrics":
        log_event(
            "http_request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


async def _timed(step: str, coro):
    start = time.perf_counter()
    try:
        return await coro
    finally:
        STEP_LATENCY.labels(step=step).observe(time.perf_counter() - start)


async def _triage(ticket: TicketInput) -> TriageResponse:
    guardrail_result = await _timed("input_guardrail", run_input_guardrail(ticket.user_message))
    for flag in guardrail_result.flags:
        GUARDRAIL_FLAGS.labels(flag=flag).inc()

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

    response = await run_orchestrator(safe_ticket, history=history, mcp_tools=app.state.mcp_tools)

    output_text = (
        response.response.draft_text
        if isinstance(response.response, ResolutionDraft)
        else response.response.summary
    )
    output_ok = await _timed("output_guardrail", run_output_guardrail(output_text))
    if not output_ok:
        OUTPUT_GUARDRAIL_BLOCKS.inc()
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


@app.post("/ticket", response_model=TriageResponse)
async def submit_ticket(ticket: TicketInput, request: Request) -> TriageResponse:
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        response = await _triage(ticket)
    latency_s = time.perf_counter() - start

    token_usage = {
        model: {k: int(u.get(k, 0)) for k in ("input_tokens", "output_tokens", "total_tokens")}
        for model, u in usage.usage_metadata.items()
    }
    response = response.model_copy(
        update={
            "request_id": request.state.request_id,
            "latency_ms": round(latency_s * 1000, 1),
            "token_usage": token_usage or None,
        }
    )

    REQUEST_LATENCY.observe(latency_s)
    FINAL_ACTIONS.labels(final_action=response.final_action).inc()
    if response.final_action == "escalated_due_to_error":
        TRANSPORT_ERRORS.inc()
    for step in response.trace:
        if step.latency_ms is not None:
            STEP_LATENCY.labels(step=step.agent_name).observe(step.latency_ms / 1000)
    for model, u in token_usage.items():
        LLM_TOKENS.labels(model=model, kind="input").inc(u["input_tokens"])
        LLM_TOKENS.labels(model=model, kind="output").inc(u["output_tokens"])

    log_event(
        "ticket_triaged",
        ticket_id=ticket.ticket_id,
        final_action=response.final_action,
        latency_ms=response.latency_ms,
        steps=[
            {"agent": s.agent_name, "latency_ms": s.latency_ms} for s in response.trace
        ],
        token_usage=token_usage,
    )
    return response
