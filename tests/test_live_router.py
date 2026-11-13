import numpy as np
import pandas as pd
import pytest

from modelrouter.gateway.app import EvalStore
from modelrouter.live import parse_letter
from modelrouter.router import DifficultyRouter, make_policy, prompt_text


def test_parse_letter():
    assert parse_letter("B") == "B"
    assert parse_letter(" c. because") == "C"
    assert parse_letter("The answer is D") == "D"
    assert parse_letter("none of them") is None


def test_prompt_text_uses_user_turns_only():
    msgs = [{"role": "system", "content": "S"}, {"role": "user", "content": "U1"}, {"role": "assistant", "content": "A"},
            {"role": "user", "content": "U2"}]
    assert prompt_text(msgs) == "U1\nU2"
    assert prompt_text([{"role": "system", "content": "only"}]) == "only"


def test_make_policy_kinds_and_errors():
    assert make_policy({"kind": "threshold", "threshold": 0.3}).threshold() == 0.3
    assert make_policy({"kind": "fraction", "target_big_fraction": 0.4}).kind == "fraction"
    assert make_policy({"kind": "budget", "budget_usd_per_1k": 1.0}, c_small=0.0001, c_big=0.002).kind == "budget"
    with pytest.raises(ValueError):
        make_policy({"kind": "nope"})


class _Const:
    def predict_proba(self, X):
        p = np.clip(X[:, 0], 0, 1)
        return np.stack([1 - p, p], axis=1)


def test_difficulty_router_threshold_for_fraction():
    bundle = {"model": _Const(), "cols": slice(None), "target": "gain", "pair": "x",
              "dev_score_quantiles": np.linspace(0, 1, 101).tolist()}
    r = DifficultyRouter(bundle)
    assert r.threshold_for_fraction(0.25) == pytest.approx(0.75)


def test_eval_store_reroutes_with_threshold(tmp_path):
    n = 10
    df = pd.DataFrame({"p_hard": np.linspace(0, 1, n), "s_score": np.r_[np.zeros(5), np.ones(5)], "b_score": np.ones(n),
                       "s_cost": np.full(n, 1.0), "b_cost": np.full(n, 10.0)})
    df.to_parquet(tmp_path / "test_predictions_x.parquet")
    (tmp_path / "offline_x.json").write_text("{}")
    s = EvalStore("x", tmp_path)
    lo, hi = s.route(0.0), s.route(1.01)
    assert lo["frac_big"] == 1.0 and lo["quality"] == 1.0 and lo["cost"] == 10.0
    assert hi["frac_big"] == 0.0 and hi["quality"] == 0.5 and hi["cost"] == 1.0
    mid = s.route(0.5)  # prompts 5..9 go big, 0..4 stay small and are wrong
    assert mid["frac_big"] == 0.5 and mid["quality"] == pytest.approx(0.5)


def test_source_prior_uses_training_rates_only():
    from modelrouter.live import _source_prior

    tr_src = np.array(["a", "a", "b", "b"])
    y = np.array([1, 1, 0, 1])
    p = _source_prior(tr_src, y, np.array(["b", "a", "a"]))
    assert p.tolist() == [0.5, 1.0, 1.0]


def test_bootstrap_interval_contains_point_estimate_for_informative_scores():
    from modelrouter import evaluate as ev
    from modelrouter.live import _bootstrap, _scores

    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.3).astype(int)
    s = (rng.random(n) < 0.5).astype(float)
    o = ev.Outcomes(s, np.maximum(s, y).astype(float), np.ones(n), np.full(n, 5.0))
    p = y * 0.6 + rng.random(n) * 0.5
    iv = _bootstrap(p, o, y, n_boot=100)
    pt = _scores(p, o, y)
    assert iv["auc"]["lo"] <= pt["auc"] <= iv["auc"]["hi"]
    assert iv["apgr"]["lo"] <= pt["apgr"] <= iv["apgr"]["hi"]
