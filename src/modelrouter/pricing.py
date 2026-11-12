"""Price table loading and cost arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class UnknownModelError(KeyError):
    pass


@dataclass(frozen=True)
class Price:
    provider: str
    input: float   # USD per 1M tokens
    output: float


class PriceTable:
    def __init__(self, models: dict[str, Price], price_date: str = "", note: str = ""):
        self.models, self.price_date, self.note = models, price_date, note

    @classmethod
    def load(cls, path: str | Path) -> PriceTable:
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        models = {}
        for name, row in (raw.get("models") or {}).items():
            inp, out = float(row["input"]), float(row["output"])
            if inp < 0 or out < 0:
                raise ValueError(f"negative price for {name}")
            models[name] = Price(str(row.get("provider", "")), inp, out)
        return cls(models, str(raw.get("price_date", "")), str(raw.get("note", "")))

    def price(self, model: str) -> Price:
        try:
            return self.models[model]
        except KeyError:
            raise UnknownModelError(f"no price for model {model!r}; add it to the pricing file") from None

    def cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        p = self.price(model)
        return (prompt_tokens * p.input + completion_tokens * p.output) / 1_000_000


def estimate_tokens(text: str) -> int:
    """Rough token estimate (about four characters per token) used only before a call returns usage."""
    return max(1, (len(text) + 3) // 4)
