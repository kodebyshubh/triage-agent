"""Session memory keyed by session_id: Redis when REDIS_URL is set, otherwise an
in-process dict.

Plain (TicketInput, TriageResponse) pairs chosen over ConversationBufferMemory -
the orchestrator's tools take a single TicketInput, not a chat-message history,
so a list of prior turns is a more direct fit than LangChain's memory classes here.

Redis backend: one list per session (`triageforge:session:<id>`), each item a
JSON turn, with the key's TTL (SESSION_TTL_S, default 24h) refreshed on every
append. That lets several backend instances share sessions and survive restarts.
In-memory backend (local dev, tests): same interface, no TTL, not persisted
across restarts - the original behavior.
"""

import json
import os

from app.schemas import TicketInput, TriageResponse

SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "86400"))
_KEY_PREFIX = "triageforge:session:"

_store: dict[str, list[tuple[TicketInput, TriageResponse]]] = {}
_redis = None


def _redis_client():
    global _redis
    url = (os.getenv("REDIS_URL") or "").strip()
    if _redis is None and url:
        import redis

        _redis = redis.Redis.from_url(url, decode_responses=True)
    return _redis


def backend() -> str:
    return "redis" if _redis_client() is not None else "memory"


def get_history(session_id: str) -> list[tuple[TicketInput, TriageResponse]]:
    r = _redis_client()
    if r is None:
        return _store.get(session_id, [])
    turns = []
    for raw in r.lrange(_KEY_PREFIX + session_id, 0, -1):
        item = json.loads(raw)
        turns.append(
            (TicketInput.model_validate(item["ticket"]), TriageResponse.model_validate(item["response"]))
        )
    return turns


def append_turn(session_id: str, ticket: TicketInput, response: TriageResponse) -> None:
    r = _redis_client()
    if r is None:
        _store.setdefault(session_id, []).append((ticket, response))
        return
    key = _KEY_PREFIX + session_id
    item = json.dumps({"ticket": ticket.model_dump(), "response": response.model_dump()})
    pipe = r.pipeline()
    pipe.rpush(key, item)
    pipe.expire(key, SESSION_TTL_S)
    pipe.execute()
