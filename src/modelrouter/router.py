"""Difficulty scoring and routing policies."""

from __future__ import annotations

import threading
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

Scorer = Callable[[str], float]


def prompt_text(messages: Sequence[dict]) -> str:
    """The text the difficulty model sees: the user turns, joined. System prompts are ignored because they
    are shared boilerplate and the classifier was trained on bare user prompts."""
    return "\n".join(m["content"] for m in messages if m.get("role") == "user") or "\n".join(
        m["content"] for m in messages)


class DifficultyRouter:
    """Wraps a trained, calibrated classifier. `score` returns the predicted probability that the big
    model is needed (higher = harder)."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.cols = bundle["cols"]
        self.target = bundle["target"]
        self.pair = bundle["pair"]
        self.quantiles = np.asarray(bundle["dev_score_quantiles"])

    @classmethod
    def load(cls, path: str | Path) -> DifficultyRouter:
        import joblib

        return cls(joblib.load(path))

    def score_many(self, prompts: Sequence[str]) -> np.ndarray:
        from .features import featurize

        X = featurize(list(prompts))
        return self.model.predict_proba(X[:, self.cols])[:, 1]

    def score(self, prompt: str) -> float:
        return float(self.score_many([prompt])[0])

    def threshold_for_fraction(self, frac: float) -> float:
        """Threshold that would send `frac` of development-set traffic to the big tier."""
        return float(np.quantile(self.quantiles, 1 - frac))


class Policy:
    """Maps a difficulty score to a tier. Subclasses differ in how the threshold is chosen."""

    kind = "base"

    def threshold(self) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def decide(self, score: float) -> str:
        return "big" if score >= self.threshold() else "small"

    def observe(self, score: float) -> None:
        pass

    def describe(self) -> dict:
        return {"kind": self.kind, "threshold": round(self.threshold(), 4)}


class ThresholdPolicy(Policy):
    kind = "threshold"

    def __init__(self, threshold: float = 0.5):
        self.t = float(threshold)

    def threshold(self) -> float:
        return self.t

    def set(self, threshold: float) -> None:
        self.t = float(threshold)


class FractionPolicy(Policy):
    """Keep the share of traffic sent to the big tier near `target` by setting the threshold to a
    quantile of recently seen scores. Before `warmup` scores exist it uses `fallback_threshold`."""

    kind = "fraction"

    def __init__(self, target: float, fallback_threshold: float = 0.5, window: int = 500, warmup: int = 30):
        if not 0 <= target <= 1:
            raise ValueError("target fraction must be within [0, 1]")
        self.target, self.fallback, self.warmup = target, fallback_threshold, warmup
        self.window: deque[float] = deque(maxlen=window)
        self._lock = threading.Lock()

    def threshold(self) -> float:
        with self._lock:
            if len(self.window) < self.warmup:
                return self.fallback
            if self.target <= 0:
                return float("inf")
            if self.target >= 1:
                return float("-inf")
            return float(np.quantile(np.fromiter(self.window, float), 1 - self.target))

    def observe(self, score: float) -> None:
        with self._lock:
            self.window.append(float(score))

    def describe(self) -> dict:
        t = self.threshold()
        return {"kind": self.kind, "target_big_fraction": self.target,
                "threshold": None if t in (float("inf"), float("-inf")) else round(t, 4),
                "window": len(self.window)}


class BudgetPolicy(FractionPolicy):
    """A spend budget (USD per 1,000 requests) turned into a target big-tier share.

    With average per-request costs c_s (small) and c_b (big), a share f costs c_s + f (c_b - c_s), so the
    largest affordable share is (budget - c_s) / (c_b - c_s), clipped to [0, 1]. The two averages are
    tracked as exponential moving averages of what requests really cost, so the share adapts."""

    kind = "budget"

    def __init__(self, usd_per_1k: float, c_small: float, c_big: float, **kw):
        self.usd_per_1k = float(usd_per_1k)
        self.c_small, self.c_big = float(c_small), float(c_big)
        super().__init__(self._share(), **kw)

    def _share(self) -> float:
        per_req = self.usd_per_1k / 1000.0
        if self.c_big <= self.c_small:
            return 0.0
        return float(np.clip((per_req - self.c_small) / (self.c_big - self.c_small), 0.0, 1.0))

    def observe_cost(self, tier: str, usd: float, alpha: float = 0.05) -> None:
        if tier == "small":
            self.c_small += alpha * (usd - self.c_small)
        else:
            self.c_big += alpha * (usd - self.c_big)
        self.target = self._share()

    def describe(self) -> dict:
        d = super().describe()
        d.update(usd_per_1k=self.usd_per_1k, c_small=self.c_small, c_big=self.c_big)
        return d


def make_policy(cfg: dict, router: DifficultyRouter | None = None, c_small: float = 0.0001,
                c_big: float = 0.002) -> Policy:
    kind = cfg.get("kind", "threshold")
    fallback = float(cfg.get("threshold", 0.5))
    if kind == "threshold":
        return ThresholdPolicy(fallback)
    if kind == "fraction":
        return FractionPolicy(float(cfg.get("target_big_fraction", 0.3)), fallback)
    if kind == "budget":
        return BudgetPolicy(float(cfg.get("budget_usd_per_1k", 1.0)), c_small, c_big, fallback_threshold=fallback)
    raise ValueError(f"unknown policy kind {kind!r}")
