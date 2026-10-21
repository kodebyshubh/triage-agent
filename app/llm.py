import asyncio
import contextvars
import itertools
import json
import os
import re
import time

import httpx
from dotenv import load_dotenv

from app.observability import RATE_LIMITED, TRANSPORT_RETRIES

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-1.5-flash"

# One place to pick the Groq model per role. Override any role with
# MODEL_<ROLE> in the environment (e.g. MODEL_ORCHESTRATOR=openai/gpt-oss-120b);
# unset roles keep the current default.
ROLES = ("classifier", "responder", "escalator", "orchestrator", "guardrail")


def role_models_from_env() -> dict[str, str]:
    return {role: (os.getenv(f"MODEL_{role.upper()}") or "").strip() or GROQ_MODEL for role in ROLES}


ROLE_MODELS = role_models_from_env()
_MAX_TRANSPORT_RETRIES = 2


_key_counter = itertools.count()


def groq_keys() -> list[str]:
    """GROQ_API_KEY plus optional GROQ_API_KEY_1..10. Empty/whitespace slots are
    skipped and duplicates dropped."""
    raw = [os.getenv("GROQ_API_KEY", "")] + [os.getenv(f"GROQ_API_KEY_{i}", "") for i in range(1, 11)]
    return list(dict.fromkeys(k.strip() for k in raw if k and k.strip()))


def next_groq_key() -> str | None:
    """Round-robin across the key pool so per-key rate limits (tokens/min,
    requests/day) are spread out. Infra only: no effect on model behavior."""
    keys = groq_keys()
    if not keys:
        return None
    return keys[next(_key_counter) % len(keys)]


# Groq rate limits are per organization and per model. When a key gets a 429,
# park that (key, model) pair for the Retry-After window and resend the same
# HTTP request with the next live key. Done at the HTTP transport so every call
# site (structured output, agent loop, guardrail) gets it without code changes,
# and it never touches what the model is asked or how its answer is handled.
_cooldown_until: dict[tuple[str, str], float] = {}
_DEFAULT_COOLDOWN_S = 60.0


def _request_model(request: httpx.Request) -> str:
    try:
        return json.loads(request.content).get("model", "")
    except (ValueError, AttributeError):
        return ""


def _retry_after(response: httpx.Response) -> float:
    try:
        return max(1.0, float(response.headers.get("retry-after", _DEFAULT_COOLDOWN_S)))
    except ValueError:
        return _DEFAULT_COOLDOWN_S


def _live_keys(model: str, tried: list[str]) -> list[str]:
    now = time.monotonic()
    return [
        k for k in groq_keys()
        if k not in tried and _cooldown_until.get((k, model), 0.0) <= now
    ]


def _pick_key(model: str, tried: list[str]) -> str | None:
    live = _live_keys(model, tried)
    return live[next(_key_counter) % len(live)] if live else None


# If every key is parked, wait for the earliest one to free up (Groq's limits
# refill continuously), up to this much total wait per request; then give up
# and return the last 429 so the caller fails visibly.
MAX_RATE_LIMIT_WAIT_S = float(os.getenv("GROQ_MAX_RATE_LIMIT_WAIT_S", "300"))


def _wait_for_key(model: str, waited: float) -> float | None:
    """Seconds to sleep before the next key frees up, or None if over budget."""
    pending = [t for (_, m), t in _cooldown_until.items() if m == model]
    if not pending:
        return None
    wait = max(0.0, min(pending) - time.monotonic()) + 0.5
    return wait if waited + wait <= MAX_RATE_LIMIT_WAIT_S else None


class _RotatingTransport(httpx.HTTPTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        model, tried, waited = _request_model(request), [], 0.0
        response = None
        while True:
            key = _pick_key(model, tried)
            if key is None:
                wait = _wait_for_key(model, waited)
                if wait is None:
                    break
                time.sleep(wait)
                waited += wait
                tried.clear()
                continue
            if response is not None:
                response.close()
            tried.append(key)
            request.headers["Authorization"] = f"Bearer {key}"
            response = super().handle_request(request)
            if response.status_code != 429:
                return response
            response.read()
            _cooldown_until[(key, model)] = time.monotonic() + _retry_after(response)
            RATE_LIMITED.labels(model=model or "unknown").inc()
        return response if response is not None else super().handle_request(request)


class _AsyncRotatingTransport(httpx.AsyncHTTPTransport):
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        model, tried, waited = _request_model(request), [], 0.0
        response = None
        while True:
            key = _pick_key(model, tried)
            if key is None:
                wait = _wait_for_key(model, waited)
                if wait is None:
                    break
                await asyncio.sleep(wait)
                waited += wait
                tried.clear()
                continue
            if response is not None:
                await response.aclose()
            tried.append(key)
            request.headers["Authorization"] = f"Bearer {key}"
            response = await super().handle_async_request(request)
            if response.status_code != 429:
                return response
            await response.aread()
            _cooldown_until[(key, model)] = time.monotonic() + _retry_after(response)
            RATE_LIMITED.labels(model=model or "unknown").inc()
        return response if response is not None else await super().handle_async_request(request)


_http_client = httpx.Client(transport=_RotatingTransport(), timeout=120)
_http_async_client = httpx.AsyncClient(transport=_AsyncRotatingTransport(), timeout=120)


# Groq error codes that mean the model's own output was malformed (unparseable tool
# call or structured JSON), not that the request or the decision was wrong. Only
# these are retried, by exact match on the error's code field.
MALFORMED_GENERATION_CODES = frozenset(
    {"tool_use_failed", "json_validate_failed", "output_parse_failed"}
)
_ERROR_CODE_RE = re.compile(r"'code': '([a-z_]+)'")


def malformed_generation_code(error: Exception) -> str | None:
    """The Groq error code if `error` is a malformed-generation failure, else None."""
    code = None
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        inner = body.get("error", body)
        code = inner.get("code") if isinstance(inner, dict) else None
    if code is None:
        match = _ERROR_CODE_RE.search(str(error))
        code = match.group(1) if match else None
    return code if code in MALFORMED_GENERATION_CODES else None


# Per-request retry tally, read back by the API layer into the response.
_retry_counts: contextvars.ContextVar[dict[str, int] | None] = contextvars.ContextVar(
    "transport_retry_counts", default=None
)


def start_retry_tracking() -> dict[str, int]:
    counts: dict[str, int] = {}
    _retry_counts.set(counts)
    return counts


def record_retry(code: str) -> None:
    TRANSPORT_RETRIES.labels(code=code).inc()
    counts = _retry_counts.get()
    if counts is not None:
        counts[code] = counts.get(code, 0) + 1


class TransportFailure(RuntimeError):
    """The model's own tool-call/structured-output generation was malformed after
    all retries - a transport-layer failure, not a content decision. Callers should
    treat this distinctly from a valid-but-disagreeable model response."""


def get_chat_model(
    temperature: float = 0, model: str | None = None, reasoning_effort: str | None = None
):
    """Groq primary, Gemini fallback (selected once per call, not retried mid-call).

    `model` overrides the default Groq model for this call only - used by the
    orchestrator to run a different model for its tool-calling loop than the one
    used for structured-output calls (classify/draft/escalate), without affecting
    those elsewhere. `reasoning_effort` is passed to Groq's gpt-oss reasoning
    models ("low"/"medium"/"high"); None keeps the model default.
    """
    groq_key = next_groq_key()
    gemini_key = os.getenv("GEMINI_API_KEY")

    if not groq_key and not gemini_key:
        raise RuntimeError(
            "Neither GROQ_API_KEY nor GEMINI_API_KEY is set. Add at least one to .env."
        )

    if groq_key:
        from langchain_groq import ChatGroq

        model = model or GROQ_MODEL
        # reasoning_effort "low"/"medium"/"high" is a gpt-oss parameter; other
        # Groq models reject it, so it is dropped for them.
        if not model.startswith("openai/gpt-oss"):
            reasoning_effort = None
        return ChatGroq(
            api_key=groq_key,
            http_client=_http_client,
            http_async_client=_http_async_client,
            model=model,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)


def invoke_structured_with_retry(structured_llm, messages):
    """Retries a with_structured_output call only on the model's own malformed
    generation (MALFORMED_GENERATION_CODES, a transport failure), never on a
    syntactically valid result - same scoping as the orchestrator's transport-retry."""
    from groq import BadRequestError

    last_error: BadRequestError | None = None
    for attempt in range(_MAX_TRANSPORT_RETRIES + 1):
        try:
            return structured_llm.invoke(messages)
        except BadRequestError as e:
            code = malformed_generation_code(e)
            if code is None:
                raise
            last_error = e
            if attempt < _MAX_TRANSPORT_RETRIES:
                record_retry(code)

    raise TransportFailure(
        f"Structured-output call failed {_MAX_TRANSPORT_RETRIES + 1} times in a row "
        "due to malformed generation from the model (transport failure, not a "
        "content disagreement)."
    ) from last_error


async def ainvoke_llm(prompt: str, temperature: float = 0) -> str:
    """Plain-text completion (used only by the guardrail checks, so it runs on the
    guardrail role's model). Falls back to Gemini if Groq raises at call time and
    a Gemini key is also configured."""
    groq_key = groq_keys()
    gemini_key = os.getenv("GEMINI_API_KEY")

    llm = get_chat_model(temperature, model=ROLE_MODELS["guardrail"])
    try:
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except Exception:
        if groq_key and gemini_key:
            from langchain_google_genai import ChatGoogleGenerativeAI

            fallback = ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)
            result = await fallback.ainvoke(prompt)
            return result.content.strip()
        raise
