"""Request IDs, structured JSON logging, Prometheus metrics and optional
LangSmith tracing.

Kept separate from the agents: nothing here changes what the model is asked or
how its answers are handled. It only records what happened.
"""

import contextvars
import json
import logging
import os
import sys
import time

from prometheus_client import Counter, Histogram

request_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)


# --- Structured JSON logging ---

class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
        }
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


logger = logging.getLogger("triageforge")


def configure_logging() -> None:
    if any(isinstance(h.formatter, JsonFormatter) for h in logger.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    logger.propagate = False


def log_event(msg: str, **fields) -> None:
    logger.info(msg, extra={"fields": fields})


# --- Prometheus metrics ---

FINAL_ACTIONS = Counter(
    "triage_final_action_total", "Tickets by final_action outcome.", ["final_action"]
)
TRANSPORT_ERRORS = Counter(
    "triage_transport_errors_total",
    "Tickets that ended in escalated_due_to_error (malformed model generation "
    "after all retries).",
)
GUARDRAIL_FLAGS = Counter(
    "triage_guardrail_flags_total", "Input guardrail flags raised.", ["flag"]
)
OUTPUT_GUARDRAIL_BLOCKS = Counter(
    "triage_output_guardrail_blocks_total", "Drafted replies blocked by the output guardrail."
)
REQUEST_ERRORS = Counter(
    "triage_request_errors_total", "Requests that raised instead of returning.", ["error_type"]
)
RATE_LIMITED = Counter(
    "triage_groq_rate_limited_total", "Groq 429 responses seen at the HTTP transport.", ["model"]
)
CLASSIFY_CACHE = Counter(
    "triage_classify_cache_total", "classify() cache lookups.", ["result"]
)
LLM_TOKENS = Counter(
    "triage_llm_tokens_total", "LLM tokens reported by the provider.", ["model", "kind"]
)
REQUEST_LATENCY = Histogram(
    "triage_request_latency_seconds",
    "End-to-end POST /ticket latency.",
    buckets=(0.5, 1, 2, 3, 5, 8, 13, 20, 30, 60, 120, 300),
)
STEP_LATENCY = Histogram(
    "triage_step_latency_seconds",
    "Latency per pipeline step (guardrails and each orchestrator tool call).",
    ["step"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 8, 13, 30, 60),
)


# --- Optional LangSmith tracing ---

def configure_tracing() -> bool:
    """Turn on LangSmith tracing only when LANGSMITH_API_KEY is set. LangChain
    reads these env vars itself; with no key, tracing is forced off so a stray
    LANGSMITH_TRACING=true can't make every call try to reach LangSmith."""
    key = (os.getenv("LANGSMITH_API_KEY") or "").strip()
    if key:
        os.environ["LANGSMITH_API_KEY"] = key  # .env values can carry stray whitespace
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ.setdefault("LANGSMITH_PROJECT", "triageforge")
        return True
    os.environ["LANGSMITH_TRACING"] = "false"
    return False
