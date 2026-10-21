"""API key auth (X-API-Key header) and a per-key rate limit for the ticket
endpoints.

TRIAGEFORGE_API_KEYS: comma-separated accepted keys. When empty, auth is off
(local dev) and the rate limit is applied per client IP instead; startup logs a
warning so this is never silent.
RATE_LIMIT_PER_MIN: requests per key per rolling minute (default 30).
RATE_LIMIT_EXEMPT_KEYS: comma-separated keys that skip the rate limit (e.g. the
eval harness key).

The rate limit uses Redis (fixed one-minute windows, shared across instances)
when REDIS_URL is set, otherwise an in-process sliding window.
"""

import hashlib
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from fastapi import Header, HTTPException, Request

from app import memory


def _csv_env(name: str) -> list[str]:
    return [k.strip() for k in os.getenv(name, "").split(",") if k.strip()]


def auth_enabled() -> bool:
    return bool(_csv_env("TRIAGEFORGE_API_KEYS"))


def _rate_limit() -> int:
    return int(os.getenv("RATE_LIMIT_PER_MIN", "30"))


_windows: dict[str, deque[float]] = defaultdict(deque)
_lock = threading.Lock()


def _hit_in_memory(identity: str, limit: int) -> int | None:
    """Records a request; returns seconds to wait if over the limit, else None."""
    now = time.monotonic()
    with _lock:
        window = _windows[identity]
        while window and window[0] <= now - 60:
            window.popleft()
        if len(window) >= limit:
            return max(1, int(60 - (now - window[0])) + 1)
        window.append(now)
    return None


def _hit_redis(r, identity: str, limit: int) -> int | None:
    minute = int(time.time() // 60)
    key = f"triageforge:ratelimit:{identity}:{minute}"
    pipe = r.pipeline()
    pipe.incr(key)
    pipe.expire(key, 61)
    count, _ = pipe.execute()
    if count > limit:
        return max(1, 60 - int(time.time() % 60))
    return None


def check_rate_limit(identity: str) -> None:
    limit = _rate_limit()
    r = memory._redis_client()
    retry_after = _hit_redis(r, identity, limit) if r is not None else _hit_in_memory(identity, limit)
    if retry_after is not None:
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded: {limit} requests per minute. Retry in {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )


async def require_api_key(
    request: Request, x_api_key: str | None = Header(default=None)
) -> str | None:
    """FastAPI dependency: 401 on a missing or wrong key, 429 over the rate limit."""
    keys = _csv_env("TRIAGEFORGE_API_KEYS")
    if keys:
        if not x_api_key:
            raise HTTPException(status_code=401, detail="Missing X-API-Key header.")
        if not any(secrets.compare_digest(x_api_key, k) for k in keys):
            raise HTTPException(status_code=401, detail="Invalid API key.")
        if x_api_key in _csv_env("RATE_LIMIT_EXEMPT_KEYS"):
            return x_api_key
        identity = "key:" + hashlib.sha256(x_api_key.encode()).hexdigest()[:16]
    else:
        identity = "ip:" + (request.client.host if request.client else "unknown")
    check_rate_limit(identity)
    return x_api_key
