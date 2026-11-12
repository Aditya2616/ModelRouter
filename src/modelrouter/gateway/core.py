"""Gateway logic, independent of the web framework."""

from __future__ import annotations

import hashlib
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from ..pricing import PriceTable
from ..providers.base import Completion, Provider, ProviderError
from ..router import BudgetPolicy, Policy, prompt_text
from .cache import ResponseCache, cache_key
from .escalation import EscalationConfig, reason_to_escalate
from .ledger import Ledger, LedgerRow


@dataclass
class Tier:
    name: str            # "small" | "big"
    provider: Provider
    model: str           # model id sent to the provider
    price_as: str        # model id in the price table used to cost this tier


@dataclass
class Gateway:
    scorer: Callable[[str], float]
    small: Tier
    big: Tier
    policy: Policy
    pricing: PriceTable
    ledger: Ledger = field(default_factory=Ledger)
    cache: ResponseCache = field(default_factory=ResponseCache)
    escalation: EscalationConfig = field(default_factory=EscalationConfig)
    max_tokens: int = 256
    temperature: float = 0.0

    def tier(self, name: str) -> Tier:
        return self.small if name == "small" else self.big

    def _cost(self, tier: Tier, c: Completion) -> float:
        return self.pricing.cost(tier.price_as, c.prompt_tokens, c.completion_tokens)

    def _pin(self, model: str, tier_override: str | None) -> str | None:
        if tier_override in ("small", "big"):
            return tier_override
        if model in ("small", "big"):
            return model
        if model == self.small.model:
            return "small"
        if model == self.big.model:
            return "big"
        return None

    def score(self, messages: list[dict]) -> dict:
        d = self.scorer(prompt_text(messages))
        t = self.policy.threshold()
        return {"difficulty": d, "threshold": t, "tier": "big" if d >= t else "small"}

    def chat(self, messages: list[dict], model: str = "auto", max_tokens: int | None = None,
             temperature: float | None = None, tier_override: str | None = None) -> dict:
        t_start = time.perf_counter()
        max_tokens = max_tokens or self.max_tokens
        temperature = self.temperature if temperature is None else temperature
        pin = self._pin(model, tier_override)
        key = cache_key(messages, max_tokens, temperature, pin)
        phash = hashlib.sha256(prompt_text(messages).encode("utf-8")).hexdigest()[:16]

        hit = self.cache.get(key)
        if hit is not None:
            resp = _copy_response(hit)
            r = resp["router"]
            self.ledger.add(LedgerRow(phash, r["difficulty"], r["tier"], resp["model"], 0, 0, 0.0, 0.0,
                                      False, True, time.perf_counter() - t_start, cache_saved_usd=r["cost_usd"]))
            r.update(cached=True, cost_usd=0.0, big_only_cost_usd=0.0, saving_usd=0.0,
                     latency_s=round(time.perf_counter() - t_start, 4))
            return resp

        difficulty = float(self.scorer(prompt_text(messages)))
        threshold = self.policy.threshold()
        self.policy.observe(difficulty)
        tier_name = pin or ("big" if difficulty >= threshold else "small")

        escalated_from: str | None = None
        reason: str | None = None
        first: Completion | None = None
        total_cost = 0.0
        tier = self.tier(tier_name)
        try:
            comp = tier.provider.complete(tier.model, messages, max_tokens, temperature)
            err: Exception | None = None
        except ProviderError as e:
            comp, err = None, e
        if comp is not None:
            total_cost += self._cost(tier, comp)
        if tier_name == "small" and pin is None:
            reason = reason_to_escalate(comp, err, self.escalation)
        if reason is not None:
            escalated_from, first = "small", comp
            tier = self.big
            tier_name = "big"
            try:
                comp = tier.provider.complete(tier.model, messages, max_tokens, temperature)
            except ProviderError as e:
                raise GatewayError(f"both tiers failed ({reason}); last error: {e}") from None
            total_cost += self._cost(tier, comp)
        elif comp is None:
            raise GatewayError(f"{tier.name} tier failed: {err}")

        # Counterfactual: the big tier alone, assuming it writes an answer of the same length.
        prompt_toks = (first or comp).prompt_tokens
        big_only = self.pricing.cost(self.big.price_as, prompt_toks, comp.completion_tokens)
        if isinstance(self.policy, BudgetPolicy):
            self.policy.observe_cost(tier_name, self._cost(tier, comp))

        latency = time.perf_counter() - t_start
        resp = {
            "id": "chatcmpl-" + uuid.uuid4().hex[:24], "object": "chat.completion", "created": int(time.time()),
            "model": comp.model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": comp.text},
                         "finish_reason": comp.finish_reason}],
            "usage": {"prompt_tokens": comp.prompt_tokens, "completion_tokens": comp.completion_tokens,
                      "total_tokens": comp.prompt_tokens + comp.completion_tokens},
            "router": {
                "difficulty": round(difficulty, 4), "threshold": None if threshold in (float("inf"), float("-inf"))
                else round(threshold, 4), "tier": tier_name, "model": comp.model, "pinned": pin is not None,
                "escalated": escalated_from is not None, "escalation_reason": reason, "cached": False,
                "cost_usd": round(total_cost, 8), "big_only_cost_usd": round(big_only, 8),
                "saving_usd": round(big_only - total_cost, 8),
                "price_basis": self.tier(tier_name).price_as, "price_date": self.pricing.price_date,
                "latency_s": round(latency, 4),
            },
        }
        self.cache.put(key, resp)
        self.ledger.add(LedgerRow(phash, difficulty, tier_name, comp.model, comp.prompt_tokens, comp.completion_tokens,
                                  total_cost, big_only, escalated_from is not None, False, latency))
        return resp


class GatewayError(RuntimeError):
    pass


def _copy_response(r: dict) -> dict:
    import copy

    return copy.deepcopy(r)
