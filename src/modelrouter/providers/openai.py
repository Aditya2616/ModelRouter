"""OpenAI chat-completions adapter (also works with any OpenAI-compatible server via base_url).

Not exercised against the real API in this repository (no key was available); covered by unit tests
that use an httpx mock transport.
"""

from __future__ import annotations

import os
import time

import httpx

from .base import Completion, Message, ProviderError, with_retries


class OpenAIProvider:
    name = "openai"

    def __init__(self, api_key: str | None = None, base_url: str = "https://api.openai.com/v1",
                 transport: httpx.BaseTransport | None = None, timeout: float = 60.0,
                 want_logprobs: bool = False, sleep=time.sleep):
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base = base_url.rstrip("/")
        self._client = httpx.Client(transport=transport, timeout=timeout)
        self._logprobs = want_logprobs
        self._sleep = sleep

    def complete(self, model: str, messages: list[Message], max_tokens: int = 256,
                 temperature: float = 0.0) -> Completion:
        if not self._key:
            raise ProviderError("OPENAI_API_KEY is not set")
        # Newer OpenAI models take max_completion_tokens; most compatible servers still use max_tokens.
        body: dict = {"model": model, "messages": messages, "temperature": temperature,
                      "max_completion_tokens": max_tokens}
        if self._logprobs:
            body["logprobs"] = True
        t0 = time.perf_counter()
        data = with_retries(lambda: self._post(body), sleep=self._sleep)
        try:
            choice = data["choices"][0]
            usage = data.get("usage") or {}
            lp = (choice.get("logprobs") or {}).get("content")
            mean_lp = sum(t["logprob"] for t in lp) / len(lp) if lp else None
            return Completion(
                text=choice["message"].get("content") or "",
                prompt_tokens=int(usage.get("prompt_tokens", 0)),
                completion_tokens=int(usage.get("completion_tokens", 0)),
                model=data.get("model", model), finish_reason=choice.get("finish_reason") or "stop",
                mean_logprob=mean_lp, latency_s=time.perf_counter() - t0, raw_id=data.get("id"),
            )
        except (KeyError, IndexError, TypeError) as e:
            raise ProviderError(f"unexpected OpenAI response shape: {type(e).__name__}") from None

    def _post(self, body: dict) -> dict:
        try:
            r = self._client.post(f"{self._base}/chat/completions", json=body,
                                  headers={"Authorization": f"Bearer {self._key}"})
        except httpx.TimeoutException:
            raise ProviderError("OpenAI request timed out", retryable=True) from None
        except httpx.HTTPError as e:
            raise ProviderError(f"OpenAI transport error: {type(e).__name__}", retryable=True) from None
        if r.status_code >= 400:
            retry = r.status_code == 429 or r.status_code >= 500
            raise ProviderError(f"OpenAI returned HTTP {r.status_code}", status=r.status_code, retryable=retry)
        return r.json()
