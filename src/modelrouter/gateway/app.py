"""FastAPI application: an OpenAI-compatible chat completions endpoint that routes by difficulty."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ..pricing import PriceTable
from ..providers import AnthropicProvider, LocalProvider, OpenAIProvider, StubProvider
from ..router import BudgetPolicy, DifficultyRouter, FractionPolicy, ThresholdPolicy, make_policy
from .cache import ResponseCache
from .core import Gateway, GatewayError, Tier
from .escalation import EscalationConfig
from .ledger import Ledger

ROOT = Path(__file__).resolve().parents[3]


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class RouterOptions(BaseModel):
    tier: Literal["small", "big", "auto"] = "auto"


class ChatRequest(BaseModel):
    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    temperature: float | None = Field(default=None, ge=0, le=2)
    stream: bool = False
    router: RouterOptions | None = None


class RouteRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=20000)


class PolicyUpdate(BaseModel):
    kind: Literal["threshold", "fraction", "budget"]
    threshold: float | None = Field(default=None, ge=0, le=1)
    target_big_fraction: float | None = Field(default=None, ge=0, le=1)
    budget_usd_per_1k: float | None = Field(default=None, gt=0)


class EvalStore:
    """The real held-out evaluation set with the router's predictions; re-routing it is plain numpy."""

    def __init__(self, pair: str, results_dir: Path):
        import pandas as pd

        self.df = pd.read_parquet(results_dir / f"test_predictions_{pair}.parquet")
        self.offline = json.loads((results_dir / f"offline_{pair}.json").read_text())
        d = self.df
        self.p, self.sq, self.bq = d.p_hard.to_numpy(float), d.s_score.to_numpy(float), d.b_score.to_numpy(float)
        self.sc, self.bc = d.s_cost.to_numpy(float), d.b_cost.to_numpy(float)

    def route(self, threshold: float) -> dict:
        big = self.p >= threshold
        n = len(self.p)
        q = np.where(big, self.bq, self.sq).mean()
        c = np.where(big, self.bc, self.sc).mean()
        return {"threshold": threshold, "n": n, "frac_big": float(big.mean()), "quality": float(q), "cost": float(c),
                "always_big_quality": float(self.bq.mean()), "always_big_cost": float(self.bc.mean()),
                "always_small_quality": float(self.sq.mean()), "always_small_cost": float(self.sc.mean()),
                "quality_retained": float(q / self.bq.mean()), "cost_vs_big": float(c / self.bc.mean()),
                "price_note": "recorded RouterBench costs at late-2023 prices"}


def _build_provider(name: str, cache: dict):
    if name not in cache:
        cache[name] = {"local": LocalProvider, "openai": OpenAIProvider, "anthropic": AnthropicProvider,
                       "stub": StubProvider}[name]()
    return cache[name]


def gateway_from_config(cfg: dict, pricing: PriceTable, router: DifficultyRouter, ledger_path: str = ":memory:",
                        providers: dict | None = None) -> Gateway:
    provs = dict(providers or {})
    tiers = {}
    for name in ("small", "big"):
        t = cfg["tiers"][name]
        tiers[name] = Tier(name, _build_provider(t["provider"], provs), t["model"], t.get("price_as", t["model"]))
    c_s = pricing.cost(tiers["small"].price_as, 200, 100)
    c_b = pricing.cost(tiers["big"].price_as, 200, 100)
    return Gateway(
        scorer=router.score, small=tiers["small"], big=tiers["big"],
        policy=make_policy(cfg.get("policy", {}), router, c_s, c_b), pricing=pricing, ledger=Ledger(ledger_path),
        cache=ResponseCache(**{k: v for k, v in (cfg.get("cache") or {}).items() if k in ("max_items", "enabled")},
                            ttl_seconds=(cfg.get("cache") or {}).get("ttl_seconds", 3600)),
        escalation=EscalationConfig.from_dict(cfg.get("escalation")),
        max_tokens=int((cfg.get("generation") or {}).get("max_tokens", 256)),
        temperature=float((cfg.get("generation") or {}).get("temperature", 0.0)),
    )


def create_app(gateway: Gateway | None = None, eval_store: EvalStore | None = None, web_dir: Path | None = None,
               config_path: str | None = None) -> FastAPI:
    app = FastAPI(title="ModelRouter", version="0.1.0", description="Difficulty-based LLM routing gateway")
    state: dict[str, Any] = {"gw": gateway, "eval": eval_store, "cfg": None}

    def gw() -> Gateway:
        if state["gw"] is None:
            cfg_file = Path(config_path or os.environ.get("MODELROUTER_CONFIG", ROOT / "config/gateway.yaml"))
            cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))
            pricing = PriceTable.load(os.environ.get("MODELROUTER_PRICING", ROOT / "config/pricing.yaml"))
            router = DifficultyRouter.load(ROOT / "models" / f"router_{cfg['router_pair']}.joblib")
            db = os.environ.get("MODELROUTER_DB", str(ROOT / "data/ledger.sqlite"))
            Path(db).parent.mkdir(exist_ok=True)
            state["gw"], state["cfg"] = gateway_from_config(cfg, pricing, router, db), cfg
        return state["gw"]

    def evals() -> EvalStore:
        if state["eval"] is None:
            pair = (state["cfg"] or {}).get("router_pair", "gpt35_vs_gpt4")
            try:
                state["eval"] = EvalStore(pair, ROOT / "results")
            except FileNotFoundError:
                raise HTTPException(404, "evaluation results not found; run `modelrouter train`") from None
        return state["eval"]

    @app.get("/health")
    def health():
        return {"status": "ok", "router_loaded": state["gw"] is not None}

    @app.get("/v1/models")
    def models():
        g = gw()
        data = [{"id": "auto", "object": "model", "owned_by": "modelrouter"}]
        data += [{"id": t.model, "object": "model", "owned_by": t.provider.name, "tier": t.name}
                 for t in (g.small, g.big)]
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    def chat(req: ChatRequest):
        g = gw()
        msgs = [m.model_dump() for m in req.messages]
        try:
            resp = g.chat(msgs, req.model, req.max_tokens, req.temperature, req.router.tier if req.router else None)
        except GatewayError as e:
            raise HTTPException(502, str(e)) from None
        if not req.stream:
            return resp
        return StreamingResponse(_sse(resp), media_type="text/event-stream")

    @app.post("/v1/route")
    def route(req: RouteRequest):
        g = gw()
        return g.score([{"role": "user", "content": req.prompt}])

    @app.get("/v1/policy")
    def get_policy():
        return gw().policy.describe()

    @app.put("/v1/policy")
    def put_policy(u: PolicyUpdate):
        g = gw()
        if u.kind == "threshold":
            if u.threshold is None:
                raise HTTPException(422, "threshold required")
            g.policy = ThresholdPolicy(u.threshold)
        elif u.kind == "fraction":
            if u.target_big_fraction is None:
                raise HTTPException(422, "target_big_fraction required")
            g.policy = FractionPolicy(u.target_big_fraction)
        else:
            if u.budget_usd_per_1k is None:
                raise HTTPException(422, "budget_usd_per_1k required")
            g.policy = BudgetPolicy(u.budget_usd_per_1k, g.pricing.cost(g.small.price_as, 200, 100),
                                    g.pricing.cost(g.big.price_as, 200, 100))
        return g.policy.describe()

    @app.get("/v1/ledger/summary")
    def ledger_summary():
        g = gw()
        return {**g.ledger.summary(), "cache": g.cache.stats(), "price_date": g.pricing.price_date}

    @app.get("/v1/ledger/recent")
    def ledger_recent(limit: int = Query(20, ge=1, le=200)):
        return gw().ledger.recent(limit)

    @app.get("/v1/eval/curve")
    def eval_curve():
        gw()
        o = evals().offline
        keys = ("pair", "small", "big", "selected", "test", "test_curve", "test_random_band", "test_oracle_curve",
                "test_advantage_vs_random")
        return {k: o[k] for k in keys if k in o}

    @app.post("/v1/eval/route")
    def eval_route(threshold: float = Query(..., ge=0, le=1)):
        gw()
        return evals().route(threshold)

    root = web_dir or ROOT / "web"
    if root.exists():
        app.mount("/", StaticFiles(directory=root, html=True), name="web")
    else:  # pragma: no cover
        @app.get("/")
        def index():
            return FileResponse(root / "index.html")

    return app


def _sse(resp: dict):
    """The gateway finishes the completion first, then replays it as server-sent events. This keeps
    OpenAI SDK streaming clients working; it is not token-by-token generation."""
    base = {"id": resp["id"], "object": "chat.completion.chunk", "created": resp["created"], "model": resp["model"]}
    text = resp["choices"][0]["message"]["content"]
    yield "data: " + json.dumps({**base, "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}) + "\n\n"
    step = 24
    for i in range(0, len(text), step):
        chunk = {**base, "choices": [{"index": 0, "delta": {"content": text[i:i + step]}, "finish_reason": None}]}
        yield "data: " + json.dumps(chunk) + "\n\n"
    last = {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": resp["choices"][0]["finish_reason"]}],
            "router": resp["router"]}
    yield "data: " + json.dumps(last) + "\n\n"
    yield "data: [DONE]\n\n"
