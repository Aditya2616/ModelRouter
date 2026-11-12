"""Anthropic Messages API adapter.

Not exercised against the real API in this repository (no key was available); covered by unit tests
that use an httpx mock transport.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import Completion, Message, ProviderError, with_retries

API_VERSION = "2023-06-01"


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.anthropic.com",
                 transport: httpx.BaseTransport | None = None, timeout: float = 60.0, sleep=time.sleep):
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=timeout)
        self._sleep = sleep

    @staticmethod
    def to_anthropic(messages: list[Message]) -> tuple[str | None, list[Message]]:
        """System text is a top-level field in this API; consecutive same-role turns are merged."""
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system") or None
        turns: list[Message] = []
        for m in messages:
            if m["role"] == "system":
                continue
            role = "assistant" if m["role"] == "assistant" else "user"
            if turns and turns[-1]["role"] == role:
                turns[-1] = {"role": role, "content": turns[-1]["content"] + "\n\n" + m["content"]}
            else:
                turns.append({"role": role, "content": m["content"]})
        return system, turns

    def complete(self, model: str, messages: list[Message], max_tokens: int = 256,
                 temperature: float = 0.0) -> Completion:
        if not self._key:
            raise ProviderError("ANTHROPIC_API_KEY is not set")
        system, turns = self.to_anthropic(messages)
        body: dict = {"model": model, "max_tokens": max_tokens, "messages": turns, "temperature": temperature}
        if system:
            body["system"] = system
        t0 = time.perf_counter()
        data = with_retries(lambda: self._post(body), sleep=self._sleep)
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            usage = data.get("usage") or {}
            stop = data.get("stop_reason") or "end_turn"
            return Completion(
                text=text, prompt_tokens=int(usage.get("input_tokens", 0)),
                completion_tokens=int(usage.get("output_tokens", 0)), model=data.get("model", model),
                finish_reason="length" if stop == "max_tokens" else "stop", mean_logprob=None,
                latency_s=time.perf_counter() - t0, raw_id=data.get("id"),
            )
        except (KeyError, TypeError) as e:
            raise ProviderError(f"unexpected Anthropic response shape: {type(e).__name__}") from None

    def _post(self, body: dict) -> dict:
        try:
            r = self._client.post(f"{self._base}/v1/messages", json=body,
                                  headers={"x-api-key": self._key, "anthropic-version": API_VERSION})
        except httpx.TimeoutException:
            raise ProviderError("Anthropic request timed out", retryable=True) from None
        except httpx.HTTPError as e:
            raise ProviderError(f"Anthropic transport error: {type(e).__name__}", retryable=True) from None
        if r.status_code >= 400:
            retry = r.status_code in (429, 529) or r.status_code >= 500
            raise ProviderError(f"Anthropic returned HTTP {r.status_code}", status=r.status_code, retryable=retry)
        return r.json()
