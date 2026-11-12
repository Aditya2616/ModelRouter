"""Exact-match response cache (LRU with TTL)."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from typing import Any


def cache_key(messages: list[dict], max_tokens: int, temperature: float, pin: str | None) -> str:
    norm = [{"role": m["role"], "content": " ".join(m["content"].split())} for m in messages]
    blob = json.dumps([norm, max_tokens, round(temperature, 4), pin], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, max_items: int = 512, ttl_seconds: float = 3600, clock: Callable[[], float] = time.monotonic,
                 enabled: bool = True):
        self.max_items, self.ttl, self.clock, self.enabled = max_items, ttl_seconds, clock, enabled
        self._d: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = self.misses = 0

    def get(self, key: str):
        if not self.enabled:
            return None
        with self._lock:
            item = self._d.get(key)
            if item is None or self.clock() - item[0] > self.ttl:
                self._d.pop(key, None)
                self.misses += 1
                return None
            self._d.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._d[key] = (self.clock(), value)
            self._d.move_to_end(key)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)

    def stats(self) -> dict:
        return {"enabled": self.enabled, "items": len(self._d), "hits": self.hits, "misses": self.misses}
