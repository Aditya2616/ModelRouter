"""Offline routing evaluation on recorded outcomes.

Every prompt has the recorded score and cost of a small and a big model, so any routing rule can be
scored exactly without calling an API. Routers rank prompts by `p_hard` (higher = send to big).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Outcomes:
    s_score: np.ndarray
    b_score: np.ndarray
    s_cost: np.ndarray
    b_cost: np.ndarray

    def __len__(self) -> int:
        return len(self.s_score)

    def take(self, idx) -> Outcomes:
        return Outcomes(self.s_score[idx], self.b_score[idx], self.s_cost[idx], self.b_cost[idx])


def curve(p_hard: np.ndarray, o: Outcomes) -> dict[str, np.ndarray]:
    """Route every prompt scoring >= t to the big model, for every distinct threshold t.

    Ties flip together, so calibrated (stepwise) scores give honest points rather than an arbitrary
    within-tie order. Returns arrays ordered from 0% to 100% of traffic sent to the big model.
    """
    n = len(p_hard)
    order = np.argsort(-p_hard, kind="stable")
    ps = p_hard[order]
    gain = (o.b_score - o.s_score)[order]
    cgain = (o.b_cost - o.s_cost)[order]
    cum_gain = np.concatenate([[0.0], np.cumsum(gain)])
    cum_cost = np.concatenate([[0.0], np.cumsum(cgain)])
    ends = np.concatenate([np.flatnonzero(ps[1:] != ps[:-1]) + 1, [n]])  # k values at tie-group ends
    ks = np.concatenate([[0], ends])
    thr = np.concatenate([[np.inf], ps[ends - 1]])
    q = (o.s_score.sum() + cum_gain[ks]) / n
    c = (o.s_cost.sum() + cum_cost[ks]) / n
    return {"frac_big": ks / n, "quality": q, "cost": c, "threshold": thr}


def interp_quality(cv: dict[str, np.ndarray], frac: np.ndarray) -> np.ndarray:
    return np.interp(frac, cv["frac_big"], cv["quality"])


def random_quality(o: Outcomes, frac: np.ndarray) -> np.ndarray:
    """Exact expectation of random routing at each fraction (a straight line)."""
    return o.s_score.mean() + frac * (o.b_score.mean() - o.s_score.mean())


def random_band(o: Outcomes, grid: np.ndarray, n_seeds: int = 200, seed: int = 0, lo=5, hi=95):
    rng = np.random.default_rng(seed)
    n = len(o)
    gain = o.b_score - o.s_score
    base = o.s_score.mean()
    out = np.empty((n_seeds, len(grid)))
    for i in range(n_seeds):
        perm = rng.permutation(n)
        cs = np.concatenate([[0.0], np.cumsum(gain[perm])])
        out[i] = base + cs[np.round(grid * n).astype(int)] / n
    return np.percentile(out, lo, axis=0), np.percentile(out, hi, axis=0)


def cpt(cv: dict[str, np.ndarray], o: Outcomes, target: float) -> float:
    """Smallest fraction of traffic to the big model reaching `target` x the big model's quality."""
    need = target * o.b_score.mean()
    ok = np.flatnonzero(cv["quality"] >= need - 1e-12)
    if len(ok) == 0:
        return float("nan")
    i = ok[0]
    if i == 0:
        return 0.0
    x0, x1 = cv["frac_big"][i - 1], cv["frac_big"][i]
    y0, y1 = cv["quality"][i - 1], cv["quality"][i]
    return float(x0 + (need - y0) / (y1 - y0) * (x1 - x0)) if y1 != y0 else float(x1)


def apgr(cv: dict[str, np.ndarray], o: Outcomes, grid_n: int = 101) -> float:
    """Average performance gap recovered (RouteLLM), averaged over call fractions 0..1."""
    grid = np.linspace(0, 1, grid_n)
    qs, qb = o.s_score.mean(), o.b_score.mean()
    if qb == qs:
        return float("nan")
    return float(np.mean((interp_quality(cv, grid) - qs) / (qb - qs)))


def _upper_hull(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = sorted(set(zip(x.tolist(), y.tolist(), strict=True)))
    hull: list[tuple[float, float]] = []
    for px, py in pts:
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (px - x1) <= (py - y1) * (x2 - x1):
                hull.pop()
            else:
                break
        hull.append((px, py))
    return np.array(hull)


def aiq(cv: dict[str, np.ndarray], o: Outcomes) -> float:
    """RouterBench-style AIQ: mean quality over the cost range [all-small, all-big], taken along the
    upper convex hull of the router's (cost, quality) points (the hull is what mixing routers can reach)."""
    h = _upper_hull(cv["cost"], cv["quality"])
    c0, c1 = o.s_cost.mean(), o.b_cost.mean()
    xs = np.linspace(c0, c1, 400)
    ys = np.interp(xs, h[:, 0], h[:, 1])
    return float(np.mean(ys))


def summarize(p_hard: np.ndarray, o: Outcomes) -> dict:
    cv = curve(p_hard, o)
    qs, qb = o.s_score.mean(), o.b_score.mean()
    return {
        "n": len(o),
        "always_small": {"quality": float(qs), "cost": float(o.s_cost.mean())},
        "always_big": {"quality": float(qb), "cost": float(o.b_cost.mean())},
        "cpt90": cpt(cv, o, 0.90), "cpt95": cpt(cv, o, 0.95), "cpt99": cpt(cv, o, 0.99),
        "apgr": apgr(cv, o), "aiq": aiq(cv, o), "aiq_random": float((qs + qb) / 2),
    }


def paired_bootstrap_advantage(p_hard: np.ndarray, o: Outcomes, fracs=(0.2, 0.3, 0.5), n_boot=500,
                               seed=0) -> dict:
    """Quality of the router minus the expected quality of random routing at the same fraction of
    big-model calls, with a bootstrap 95% interval over prompts."""
    rng = np.random.default_rng(seed)
    n = len(o)
    res: dict[float, list[float]] = {f: [] for f in fracs}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        ob, pb = o.take(idx), p_hard[idx]
        cv = curve(pb, ob)
        for f in fracs:
            fa = np.array([f])
            res[f].append(float(interp_quality(cv, fa)[0] - random_quality(ob, fa)[0]))
    return {
        f"{int(f * 100)}%": {"mean": float(np.mean(v)), "lo": float(np.percentile(v, 2.5)),
                             "hi": float(np.percentile(v, 97.5))}
        for f, v in res.items()
    }


def downsample(cv: dict[str, np.ndarray], max_pts: int = 240) -> dict[str, list]:
    n = len(cv["frac_big"])
    idx = np.unique(np.linspace(0, n - 1, min(n, max_pts)).astype(int))
    return {k: np.round(v[idx], 6).tolist() for k, v in cv.items() if k != "threshold"}
