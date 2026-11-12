import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from modelrouter.gateway.app import create_app
from modelrouter.gateway.cache import ResponseCache, cache_key
from modelrouter.gateway.core import Gateway, GatewayError, Tier
from modelrouter.gateway.escalation import EscalationConfig
from modelrouter.gateway.ledger import Ledger
from modelrouter.pricing import PriceTable
from modelrouter.providers import StubProvider
from modelrouter.router import BudgetPolicy, FractionPolicy, ThresholdPolicy


@pytest.fixture
def pricing():
    return PriceTable.load("config/pricing.yaml")


def make_gateway(pricing, score=0.2, policy=None, small=None, big=None, escalation=None, cache=None):
    small = small or StubProvider(lambda m, msgs: "small answer", mean_logprob=-0.1)
    big = big or StubProvider(lambda m, msgs: "big answer", mean_logprob=-0.1)
    scorer = score if callable(score) else (lambda text: score)
    gw = Gateway(scorer=scorer, small=Tier("small", small, "stub-small", "stub-small"),
                 big=Tier("big", big, "stub-big", "stub-big"), policy=policy or ThresholdPolicy(0.5),
                 pricing=pricing, ledger=Ledger(), cache=cache or ResponseCache(),
                 escalation=escalation or EscalationConfig())
    return gw, small, big


MSG = [{"role": "user", "content": "What is 2+2?"}]


def test_easy_goes_small_hard_goes_big(pricing):
    gw, small, big = make_gateway(pricing, score=0.2)
    r = gw.chat(MSG)
    assert r["router"]["tier"] == "small" and r["choices"][0]["message"]["content"] == "small answer"
    assert len(small.calls) == 1 and len(big.calls) == 0
    gw2, small2, big2 = make_gateway(pricing, score=0.9)
    r2 = gw2.chat(MSG)
    assert r2["router"]["tier"] == "big" and len(big2.calls) == 1 and len(small2.calls) == 0


def test_cost_and_saving_metadata(pricing):
    gw, *_ = make_gateway(pricing, score=0.1)
    r = gw.chat(MSG)["router"]
    assert r["big_only_cost_usd"] > r["cost_usd"] > 0
    assert r["saving_usd"] == pytest.approx(r["big_only_cost_usd"] - r["cost_usd"], abs=1e-9)
    hard, *_ = make_gateway(pricing, score=0.99)
    h = hard.chat(MSG)["router"]
    assert h["saving_usd"] == pytest.approx(0.0, abs=1e-12)


def test_pinning_by_model_and_override(pricing):
    gw, small, big = make_gateway(pricing, score=0.1)
    assert gw.chat(MSG, model="big")["router"]["tier"] == "big"
    assert gw.chat([{"role": "user", "content": "x"}], model="auto", tier_override="big")["router"]["pinned"] is True
    assert len(big.calls) == 2 and len(small.calls) == 0


def test_escalates_on_small_error_and_books_only_paid_calls(pricing):
    gw, small, big = make_gateway(pricing, small=StubProvider(fail=True))
    r = gw.chat(MSG)
    assert r["router"]["escalated"] and r["router"]["escalation_reason"] == "error" and r["router"]["tier"] == "big"
    assert r["choices"][0]["message"]["content"] == "big answer"


def test_escalates_on_empty_and_low_confidence_and_books_both_costs(pricing):
    gw, *_ = make_gateway(pricing, small=StubProvider(lambda m, x: "   ", mean_logprob=-0.1))
    r = gw.chat(MSG)
    assert r["router"]["escalation_reason"] == "empty"
    gw2, *_ = make_gateway(pricing, small=StubProvider(lambda m, x: "maybe", mean_logprob=-3.0))
    r2 = gw2.chat(MSG)
    assert r2["router"]["escalation_reason"] == "low_confidence"
    plain, *_ = make_gateway(pricing, score=0.99)
    only_big = plain.chat(MSG)["router"]["cost_usd"]
    assert r2["router"]["cost_usd"] > only_big  # the wasted small call is paid for as well


def test_no_escalation_when_pinned_or_disabled(pricing):
    gw, _, big = make_gateway(pricing, small=StubProvider(fail=True), escalation=EscalationConfig(enabled=False))
    with pytest.raises(GatewayError):
        gw.chat(MSG)
    assert len(big.calls) == 0


def test_both_tiers_failing_raises(pricing):
    gw, *_ = make_gateway(pricing, small=StubProvider(fail=True), big=StubProvider(fail=True))
    with pytest.raises(GatewayError):
        gw.chat(MSG)


def test_cache_hit_is_free_and_does_not_call_providers(pricing):
    gw, small, _ = make_gateway(pricing, score=0.1)
    first = gw.chat(MSG)
    second = gw.chat([{"role": "user", "content": "What  is 2+2?"}])  # whitespace-normalised key
    assert second["router"]["cached"] and second["router"]["cost_usd"] == 0.0
    assert second["choices"][0]["message"]["content"] == first["choices"][0]["message"]["content"]
    assert len(small.calls) == 1
    s = gw.ledger.summary()
    assert s["requests"] == 2 and s["cache_hits"] == 1 and s["cache_saved_usd"] > 0
    # routing saving only counts the live request
    assert s["routing_saving_usd"] == pytest.approx(first["router"]["saving_usd"], abs=1e-9)


def test_cache_ttl_and_lru():
    t = [0.0]
    c = ResponseCache(max_items=2, ttl_seconds=10, clock=lambda: t[0])
    c.put("a", 1)
    c.put("b", 2)
    c.put("c", 3)
    assert c.get("a") is None and c.get("b") == 2
    t[0] = 11
    assert c.get("b") is None
    assert cache_key(MSG, 10, 0.0, None) != cache_key(MSG, 10, 0.0, "big")


def test_ledger_never_stores_prompt_text(pricing):
    gw, *_ = make_gateway(pricing)
    gw.chat([{"role": "user", "content": "my secret prompt sk-123"}])
    rows = gw.ledger.recent()
    assert "secret" not in json.dumps(rows) and len(rows[0]["prompt_hash"]) == 16


def test_fraction_policy_tracks_target():
    rng = np.random.default_rng(0)
    pol = FractionPolicy(0.3, window=400, warmup=30)
    picks = []
    for s in rng.beta(2, 3, 3000):
        picks.append(pol.decide(s) == "big")
        pol.observe(s)
    assert abs(np.mean(picks[500:]) - 0.3) < 0.05


def test_budget_policy_share_and_limits():
    pol = BudgetPolicy(usd_per_1k=1.0, c_small=0.0002, c_big=0.002)  # $0.001 per request
    assert pol.target == pytest.approx((0.001 - 0.0002) / (0.002 - 0.0002))
    assert BudgetPolicy(100.0, 0.0002, 0.002).target == 1.0
    assert BudgetPolicy(0.01, 0.0002, 0.002).target == 0.0
    rng = np.random.default_rng(1)
    spend, n = 0.0, 4000
    for s in rng.random(n):
        tier = pol.decide(s)
        pol.observe(s)
        cost = 0.002 if tier == "big" else 0.0002
        spend += cost
        pol.observe_cost(tier, cost)
    assert spend / n <= 0.001 * 1.08


def test_threshold_monotonic(pricing):
    scores = np.random.default_rng(0).random(300)
    shares = []
    for t in (0.2, 0.5, 0.8):
        pol = ThresholdPolicy(t)
        shares.append(np.mean([pol.decide(s) == "big" for s in scores]))
    assert shares[0] > shares[1] > shares[2]


def client(pricing, **kw):
    gw, small, big = make_gateway(pricing, **kw)
    return TestClient(create_app(gw, web_dir=None)), gw, small, big


def test_health_and_models(pricing):
    c, *_ = client(pricing)
    assert c.get("/health").json()["status"] == "ok"
    ids = [m["id"] for m in c.get("/v1/models").json()["data"]]
    assert ids == ["auto", "stub-small", "stub-big"]


def test_chat_completions_schema_and_router_block(pricing):
    c, *_ = client(pricing, score=0.8)
    r = c.post("/v1/chat/completions", json={"messages": MSG})
    assert r.status_code == 200
    j = r.json()
    assert j["object"] == "chat.completion" and j["choices"][0]["message"]["role"] == "assistant"
    assert set(j["router"]) >= {"difficulty", "tier", "model", "cost_usd", "big_only_cost_usd", "saving_usd", "escalated"}
    assert j["router"]["tier"] == "big" and j["usage"]["total_tokens"] > 0


def test_streaming_returns_sse_chunks(pricing):
    c, *_ = client(pricing)
    r = c.post("/v1/chat/completions", json={"messages": MSG, "stream": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    lines = [ln for ln in r.text.split("\n\n") if ln.startswith("data: ")]
    assert lines[-1] == "data: [DONE]"
    chunks = [json.loads(ln[6:]) for ln in lines[:-1]]
    text = "".join(ch["choices"][0]["delta"].get("content", "") for ch in chunks)
    assert text == "small answer"


def test_bad_input_is_422(pricing):
    c, *_ = client(pricing)
    assert c.post("/v1/chat/completions", json={"messages": []}).status_code == 422
    assert c.post("/v1/chat/completions", json={"messages": [{"role": "x", "content": "a"}]}).status_code == 422
    assert c.post("/v1/chat/completions", json={"messages": MSG, "temperature": 9}).status_code == 422


def test_upstream_failure_is_502(pricing):
    c, *_ = client(pricing, small=StubProvider(fail=True), big=StubProvider(fail=True))
    assert c.post("/v1/chat/completions", json={"messages": MSG}).status_code == 502


def test_route_and_policy_endpoints(pricing):
    c, *_ = client(pricing, score=0.7)
    r = c.post("/v1/route", json={"prompt": "hello"}).json()
    assert r["difficulty"] == 0.7 and r["tier"] == "big"
    assert c.put("/v1/policy", json={"kind": "threshold", "threshold": 0.9}).json()["threshold"] == 0.9
    assert c.post("/v1/route", json={"prompt": "hello"}).json()["tier"] == "small"
    assert c.put("/v1/policy", json={"kind": "threshold"}).status_code == 422
    assert c.put("/v1/policy", json={"kind": "budget", "budget_usd_per_1k": 0.5}).json()["kind"] == "budget"


def test_ledger_summary_endpoint(pricing):
    c, *_ = client(pricing, score=0.1)
    c.post("/v1/chat/completions", json={"messages": MSG})
    s = c.get("/v1/ledger/summary").json()
    assert s["requests"] == 1 and s["routing_saving_usd"] > 0 and s["price_date"] == "2026-11-12"
