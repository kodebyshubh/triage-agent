import itertools
import os

from dotenv import load_dotenv

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-20b"
GEMINI_MODEL = "gemini-1.5-flash"
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

        return ChatGroq(
            api_key=groq_key,
            model=model or GROQ_MODEL,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(model=GEMINI_MODEL, temperature=temperature)


def invoke_structured_with_retry(structured_llm, messages):
    """Retries a with_structured_output call only on the model's own malformed
    tool-call generation (a transport failure), never on a syntactically valid
    result - same scoping as the orchestrator's transport-retry."""
    from groq import BadRequestError

    last_error: BadRequestError | None = None
    for _ in range(_MAX_TRANSPORT_RETRIES + 1):
        try:
            return structured_llm.invoke(messages)
        except BadRequestError as e:
            if "tool_use_failed" not in str(e):
                raise
            last_error = e

    raise TransportFailure(
        f"Structured-output call failed {_MAX_TRANSPORT_RETRIES + 1} times in a row "
        "due to malformed generation from the model (transport failure, not a "
        "content disagreement)."
    ) from last_error


async def ainvoke_llm(prompt: str, temperature: float = 0) -> str:
    """Plain-text completion. Falls back to Gemini if Groq raises at call time
    and a Gemini key is also configured."""
    groq_key = groq_keys()
    gemini_key = os.getenv("GEMINI_API_KEY")

    llm = get_chat_model(temperature)
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
