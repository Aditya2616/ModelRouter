"""Deterministic in-memory provider for tests and for running the gateway without any model."""

from __future__ import annotations

from collections.abc import Callable

from ..pricing import estimate_tokens
from .base import Completion, Message, ProviderError


class StubProvider:
    name = "stub"

    def __init__(self, reply: Callable[[str, list[Message]], str] | None = None,
                 mean_logprob: float | None = -0.2, fail: bool = False):
        self.reply = reply or (lambda model, msgs: f"[{model}] " + msgs[-1]["content"][:40])
        self.mean_logprob = mean_logprob
        self.fail = fail
        self.calls: list[tuple[str, list[Message]]] = []

    def complete(self, model: str, messages: list[Message], max_tokens: int = 256,
                 temperature: float = 0.0) -> Completion:
        self.calls.append((model, messages))
        if self.fail:
            raise ProviderError("stub failure", retryable=False)
        text = self.reply(model, messages)
        return Completion(
            text=text, prompt_tokens=sum(estimate_tokens(m["content"]) for m in messages),
            completion_tokens=estimate_tokens(text), model=model, mean_logprob=self.mean_logprob,
        )
