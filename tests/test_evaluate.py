import numpy as np

from modelrouter import evaluate as ev


def make(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    s = (rng.random(n) < 0.6).astype(float)
    b = np.where(s == 1, 1.0, (rng.random(n) < 0.5).astype(float))
    return ev.Outcomes(s, b, np.full(n, 1.0), np.full(n, 10.0))


def test_endpoints_match_always_small_and_big():
    o = make()
    cv = ev.curve(np.random.default_rng(1).random(len(o)), o)
    assert cv["frac_big"][0] == 0 and cv["frac_big"][-1] == 1
    assert np.isclose(cv["quality"][0], o.s_score.mean())
    assert np.isclose(cv["quality"][-1], o.b_score.mean())
    assert np.isclose(cv["cost"][0], 1.0) and np.isclose(cv["cost"][-1], 10.0)


def test_oracle_dominates_and_random_is_a_line():
    o = make()
    oracle = ev.curve(o.b_score - o.s_score, o)
    rnd = ev.random_quality(o, oracle["frac_big"])
    assert np.all(oracle["quality"] >= rnd - 1e-9)
    wins = (o.b_score > o.s_score).mean()
    assert ev.cpt(oracle, o, 1.0) <= wins + 1e-6
    half = ev.random_quality(o, np.array([0.5]))[0]
    assert np.isclose(half, (o.s_score.mean() + o.b_score.mean()) / 2)


def test_random_scores_track_the_random_line():
    o = make(5000)
    p = np.random.default_rng(3).random(len(o))
    cv = ev.curve(p, o)
    grid = np.linspace(0, 1, 21)
    gap = np.abs(ev.interp_quality(cv, grid) - ev.random_quality(o, grid))
    assert gap.max() < 0.03
    assert abs(ev.apgr(cv, o) - 0.5) < 0.05


def test_ties_flip_together():
    o = make(100)
    cv = ev.curve(np.zeros(100), o)
    assert list(cv["frac_big"]) == [0.0, 1.0]


def test_cpt_and_apgr_hand_case():
    o = ev.Outcomes(np.array([0, 0, 1, 1.0]), np.array([1, 1, 1, 1.0]), np.ones(4), np.full(4, 5.0))
    cv = ev.curve(np.array([0.9, 0.8, 0.2, 0.1]), o)
    assert np.isclose(ev.cpt(cv, o, 1.0), 0.5)
    assert ev.apgr(cv, o) > 0.7


def test_aiq_better_router_scores_higher():
    o = make()
    jitter = 1e-3 * np.random.default_rng(0).random(len(o))
    good = ev.aiq(ev.curve(o.b_score - o.s_score + jitter, o), o)
    rnd = ev.aiq(ev.curve(np.random.default_rng(0).random(len(o)), o), o)
    assert good > rnd


def test_bootstrap_advantage_positive_for_oracle():
    o = make(500)
    adv = ev.paired_bootstrap_advantage(o.b_score - o.s_score, o, n_boot=50)
    assert adv["30%"]["lo"] > 0
