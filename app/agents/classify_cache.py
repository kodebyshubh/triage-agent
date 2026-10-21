"""Small TTL + LRU cache for classify() on identical messages.

classify() depends only on the message text and the classifier model, so an
identical message (a resubmit, a duplicate ticket) can reuse the earlier result
instead of paying another LLM round trip. Only successful results are cached;
a TransportFailure is raised as usual and never stored.

CLASSIFY_CACHE_SIZE (default 256; 0 disables) and CLASSIFY_CACHE_TTL_S
(default 600). The eval harness runs with the cache disabled, so repeated runs
of the same ticket stay independent samples.
"""

import functools
import os
import threading
import time
from collections import OrderedDict

from app.observability import CLASSIFY_CACHE

_lock = threading.Lock()
_entries: OrderedDict = OrderedDict()


def _size() -> int:
    return int(os.getenv("CLASSIFY_CACHE_SIZE", "256"))


def _ttl() -> float:
    return float(os.getenv("CLASSIFY_CACHE_TTL_S", "600"))


def clear() -> None:
    with _lock:
        _entries.clear()


def cached(classify_fn, model_for_key):
    @functools.wraps(classify_fn)
    def wrapper(ticket):
        size = _size()
        if size <= 0:
            return classify_fn(ticket)
        key = (model_for_key(), ticket.user_message)
        now = time.monotonic()
        with _lock:
            hit = _entries.get(key)
            if hit is not None and hit[0] > now:
                _entries.move_to_end(key)
                CLASSIFY_CACHE.labels(result="hit").inc()
                return hit[1].model_copy()
        CLASSIFY_CACHE.labels(result="miss").inc()
        result = classify_fn(ticket)
        with _lock:
            _entries[key] = (now + _ttl(), result)
            _entries.move_to_end(key)
            while len(_entries) > size:
                _entries.popitem(last=False)
        return result.model_copy()

    return wrapper
