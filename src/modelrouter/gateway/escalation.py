"""When should an answer from the small tier be thrown away and re-run on the big tier?"""

from __future__ import annotations

from dataclasses import dataclass

from ..providers.base import Completion

REFUSAL_MARKERS = ("i cannot", "i can't", "i'm sorry, but", "i am unable", "as an ai", "i don't know")


@dataclass
class EscalationConfig:
    enabled: bool = True
    min_mean_logprob: float | None = -1.6
    escalate_on_empty: bool = True
    escalate_on_error: bool = True
    escalate_on_truncation: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> EscalationConfig:
        d = d or {}
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def reason_to_escalate(c: Completion | None, error: Exception | None, cfg: EscalationConfig) -> str | None:
    """Return a short reason string, or None to accept the small tier's answer."""
    if not cfg.enabled:
        return None
    if error is not None:
        return "error" if cfg.escalate_on_error else None
    assert c is not None
    if cfg.escalate_on_empty and not c.text.strip():
        return "empty"
    if cfg.escalate_on_truncation and c.finish_reason == "length":
        return "truncated"
    if c.mean_logprob is not None and cfg.min_mean_logprob is not None and c.mean_logprob < cfg.min_mean_logprob:
        return "low_confidence"
    head = c.text.strip().lower()[:80]
    if any(head.startswith(m) for m in REFUSAL_MARKERS):
        return "refusal"
    return None
