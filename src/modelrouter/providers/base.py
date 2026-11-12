"""One interface for every model backend."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

Message = dict[str, str]


class ProviderError(RuntimeError):
    """Raised for any backend failure. Never includes credentials."""

    def __init__(self, message: str, status: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str
    finish_reason: str = "stop"
    mean_logprob: float | None = None   # mean log-probability of generated tokens, when the backend gives it
    latency_s: float = 0.0
    raw_id: str | None = None


class Provider(Protocol):
    name: str

    def complete(self, model: str, messages: list[Message], max_tokens: int = 256,
                 temperature: float = 0.0) -> Completion: ...


def with_retries(fn: Callable[[], Any], attempts: int = 3, base_delay: float = 0.5,
                 sleep: Callable[[float], None] = time.sleep) -> Any:
    """Retry on retryable ProviderErrors (429 / 5xx / timeouts) with exponential backoff."""
    last: ProviderError | None = None
    for i in range(attempts):
        try:
            return fn()
        except ProviderError as e:
            last = e
            if not e.retryable or i == attempts - 1:
                raise
            sleep(base_delay * (2 ** i))
    raise last  # pragma: no cover
