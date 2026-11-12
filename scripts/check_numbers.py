"""Keep every number in README.md, the web data and the pricing config in step with the results files.

    python scripts/check_numbers.py --write   # regenerate the marked parts of README.md
    python scripts/check_numbers.py --check   # exit 1 if anything is out of sync (also run by a fast test)

README.md marks generated text in two ways:
    <!--BEGIN:name--> ... <!--END:name-->   a whole table or paragraph, rendered from results/*.json
    <!--v:name-->value<!--/v-->             one inline number in prose

Everything comes from results/offline_*.json, results/live.json, results/test_predictions_*.parquet and
config/pricing.yaml. The check also asserts the claims the prose makes (for example "every interval includes
zero"), and that web/data and docs/data hold exactly the exported results.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modelrouter.features import LEXICAL_NAMES  # noqa: E402

PAIRS = ("gpt35_vs_gpt4", "mixtral_vs_gpt4")


def f3(x: float) -> str:
    return f"{x:.3f}"


def sg(x: float, d: int = 3) -> str:
    return f"{x:+.{d}f}"


def pc(x: float, d: int = 0) -> str:
    return f"{100 * x:.{d}f}%"


def ci(a: dict) -> str:
    return f"{sg(a['mean'])} [{sg(a['lo'])}, {sg(a['hi'])}]"


def load() -> dict:
    off = {p: json.loads((ROOT / "results" / f"offline_{p}.json").read_text(encoding="utf-8")) for p in PAIRS}
    pred = {p: pd.read_parquet(ROOT / "results" / f"test_predictions_{p}.parquet") for p in PAIRS}
    live = json.loads((ROOT / "results" / "live.json").read_text(encoding="utf-8"))
    pricing = yaml.safe_load((ROOT / "config" / "pricing.yaml").read_text(encoding="utf-8"))
    return {"off": off, "pred": pred, "live": live, "pricing": pricing}


def operating_point(pred: pd.DataFrame, frac: float) -> dict:
    """Quality and cost at the UI's "Match big 95%" preset: threshold = difficulty of the k-th hardest prompt with
    k = round(frac * n), and every prompt with difficulty >= threshold goes to the big model (ties included)."""
    p = pred.p_hard.to_numpy()
    n = len(pred)
    k0 = round(frac * n)
    t = np.sort(p)[::-1][k0 - 1]
    big = p >= t
    k = int(big.sum())
    q = (np.where(big, pred.b_score, pred.s_score)).sum() / n
    c = (np.where(big, pred.b_cost, pred.s_cost)).sum() / n
    return {"k": k, "n": n, "frac": k / n, "quality": q, "quality_vs_big": q / pred.b_score.mean(),
            "cost_vs_big": c / pred.b_cost.mean(), "cost_per_1k": c * 1000, "big_cost_per_1k": pred.b_cost.mean() * 1000}


def values(D: dict) -> tuple[dict[str, str], dict[str, str]]:
    off, pred, live, pr = D["off"], D["pred"], D["live"], D["pricing"]
    v: dict[str, str] = {}
    blocks: dict[str, str] = {}
    a, b = off[PAIRS[0]], off[PAIRS[1]]
    ta, tb = a["test"], b["test"]
    v.update(n_total=f"{a['n_total']:,}", n_dev=f"{a['n_dev']:,}", n_test=f"{a['n_test']:,}",
             n_models="11", n_lexical=str(len(LEXICAL_NAMES)), small_a=a["small"], big_a=a["big"], small_b=b["small"], big_b=b["big"])
    assert a["n_dev"] + a["n_test"] == a["n_total"] and sum(a["families"].values()) == a["n_total"]
    assert (a["n_total"], a["n_dev"], a["n_test"]) == (b["n_total"], b["n_dev"], b["n_test"])
    assert len(pred[PAIRS[0]]) == a["n_test"]
    sel = a["selected"]
    assert sel["model"] == "rf" and sel["target"] == "small_fail" and sel == b["selected"]
    v["classifier"] = "calibrated random forest predicting that the small model gets the prompt wrong"
    # pair A headline
    oa = operating_point(pred[PAIRS[0]], ta["cpt95"])
    ob = operating_point(pred[PAIRS[1]], tb["cpt95"])
    v.update(a_small_q=f3(ta["always_small"]["quality"]), a_big_q=f3(ta["always_big"]["quality"]),
             a_auc=f3(ta["auc"]), a_apgr=f3(ta["apgr"]), a_aiq=f3(ta["aiq"]), a_aiq_rand=f3(ta["aiq_random"]),
             a_cpt90=pc(ta["cpt90"]), a_cpt95=pc(ta["cpt95"]), a_op_frac=pc(oa["frac"], 1), a_cpt99=pc(ta["cpt99"]),
             a_op_cost=pc(oa["cost_vs_big"]), a_op_saving=pc(1 - oa["cost_vs_big"]), a_op_quality=pc(oa["quality_vs_big"], 1),
             a_op_k=f"{oa['k']:,}", a_op_per1k=f"${oa['cost_per_1k']:.2f}", a_big_per1k=f"${oa['big_cost_per_1k']:.2f}",
             a_adv30=ci(a["test_advantage_vs_random"]["30%"]), a_oracle_cpt95=pc(a["test_oracle"]["cpt95"]),
             a_cost_ratio=f"{ta['always_big']['cost'] / ta['always_small']['cost']:.1f}x",
             b_small_q=f3(tb["always_small"]["quality"]), b_auc=f3(tb["auc"]), b_apgr=f3(tb["apgr"]),
             b_cpt95=pc(tb["cpt95"]), b_op_cost=pc(ob["cost_vs_big"]), b_op_quality=pc(ob["quality_vs_big"], 1),
             b_adv30=ci(b["test_advantage_vs_random"]["30%"]))
    assert 0.5 < ta["apgr"] and a["test_advantage_vs_random"]["30%"]["lo"] > 0 and b["test_advantage_vs_random"]["30%"]["lo"] > 0
    blocks["offline_a"] = "\n".join([
        "| Held-out test, pair A | value |", "|---|---|",
        f"| always-small quality / always-big quality | {f3(ta['always_small']['quality'])} / {f3(ta['always_big']['quality'])} |",
        f"| AUC of the classifier | {f3(ta['auc'])} |",
        f"| APGR (random routing is 0.5) | {f3(ta['apgr'])} |",
        f"| AIQ (random routing: {f3(ta['aiq_random'])}) | {f3(ta['aiq'])} |",
        f"| CPT 90% / 95% / 99% (share of calls to the big model needed to keep that share of its quality) | "
        f"{pc(ta['cpt90'])} / {pc(ta['cpt95'])} / {pc(ta['cpt99'])} |",
        f"| at the 95% operating point ({oa['k']:,} of {oa['n']:,} prompts to big) | quality {pc(oa['quality_vs_big'], 1)} of big, "
        f"cost {pc(oa['cost_vs_big'])} of all-big ({pc(1 - oa['cost_vs_big'])} cheaper; ${oa['cost_per_1k']:.2f} vs "
        f"${oa['big_cost_per_1k']:.2f} per 1,000 requests at Nov 2023 prices) |",
        f"| advantage over random routing at 30% to big (95% bootstrap interval) | {ci(a['test_advantage_vs_random']['30%'])} |",
        f"| oracle CPT 95% (upper bound) | {pc(a['test_oracle']['cpt95'])} |",
    ])
    # cv table
    rows = ["| model | AUC | Brier | ECE | APGR | CPT 95% |", "|---|---|---|---|---|---|"]
    names = [("lr_lex", "logistic regression, lexical features only"), ("lr", "logistic regression, embeddings + lexical"),
             ("hgb", "HistGradientBoosting"), ("rf", "random forest (selected)"), ("mlp", "MLP"),
             ("task_prior", "baseline: sub-benchmark identity only"), ("family_prior", "baseline: benchmark family only")]
    for key, label in names:
        m = a["cv"][f"small_fail/{key}"]
        rows.append(f"| {label} | {f3(m['auc'])} | {f3(m['brier'])} | {f3(m['ece'])} | {f3(m['apgr'])} | {pc(m['cpt95'])} |")
    blocks["cv_a"] = "\n".join(rows)
    cv = a["cv"]
    best_cv = max((k for k in cv if k.split("/")[0] == "small_fail" and k.split("/")[1] in ("lr_lex", "lr", "hgb", "rf", "mlp")),
                  key=lambda k: cv[k]["apgr"])
    assert best_cv == "small_fail/rf"
    v.update(cv_rf_apgr=f3(cv["small_fail/rf"]["apgr"]), cv_task_apgr=f3(cv["small_fail/task_prior"]["apgr"]),
             cv_fam_apgr=f3(cv["small_fail/family_prior"]["apgr"]), test_fam_apgr=f3(a["test_family_prior"]["apgr"]))
    assert cv["small_fail/task_prior"]["apgr"] > cv["small_fail/family_prior"]["apgr"]
    # leave-family-out tables
    for tag, o in (("a", a), ("b", b)):
        rows = ["| held-out family | n | AUC | APGR | advantage vs random at 30% [95% CI] |", "|---|---|---|---|---|"]
        for fam, m in o["leave_family_out"].items():
            rows.append(f"| {fam} | {m['n']:,} | {f3(m['auc']) if 'auc' in m else 'n/a'} | {f3(m['apgr'])} | "
                        f"{ci(m['advantage_vs_random_30pct'])} |")
        blocks[f"lofo_{tag}"] = "\n".join(rows)
        L = o["leave_family_out"]
        aucs = [m["auc"] for m in L.values() if "auc" in m]
        advs = [m["advantage_vs_random_30pct"] for m in L.values()]
        sig = sum(x["lo"] > 0 for x in advs)
        v.update({f"{tag}_lofo_nfam": str(len(L)), f"{tag}_lofo_auc_lt60": str(sum(x < 0.6 for x in aucs)),
                  f"{tag}_lofo_max_adv": sg(max(x["mean"] for x in advs)), f"{tag}_lofo_sig": str(sig),
                  f"{tag}_lofo_mean_apgr": f3(sum(m["apgr"] for m in L.values()) / len(L)),
                  f"{tag}_lofo_n": f"{sum(m['n'] for m in L.values()):,}"})
    assert max(x["mean"] for x in [m["advantage_vs_random_30pct"] for m in a["leave_family_out"].values()]) < 0.02
    v["b_cv_rf_apgr"] = f3(b["cv"]["small_fail/rf"]["apgr"])
    # live
    n = live["n"]
    acc, bs = live["accuracy"], live["by_source"]
    assert sum(s["n"] for s in bs.values()) == n == live["per_source"] * len(bs)
    run = live["run"]
    cr = {k: live[k] for k in ("router_cv", "source_prior", "routerbench_transfer")}
    assert all(c["advantage_vs_random"]["30%"]["lo"] < 0 < c["advantage_vs_random"]["30%"]["hi"] for c in cr.values()), \
        "README says every live interval includes zero"
    v.update(live_n=str(n), live_per=str(live["per_source"]), live_small=live["small"].split("/")[1],
             live_big=live["big"].split("/")[1], live_acc_s=pc(acc["small"], 1), live_acc_b=pc(acc["big"], 1),
             live_acc_o=pc(acc["oracle"], 1), live_strict=pc(live["share_big_strictly_better"]),
             live_strict_n=str(round(live["share_big_strictly_better"] * n)),
             live_lat_s=f"{live['mean_latency_s']['small']:.2f}", live_lat_b=f"{live['mean_latency_s']['big']:.2f}",
             live_tps_s=f"{live['prompt_tokens_per_s']['small']:.0f}", live_tps_b=f"{live['prompt_tokens_per_s']['big']:.0f}",
             live_ptok=f"{live['mean_prompt_tokens']['small']:.0f}", live_dev=f"{run['gpu']} ({run['device']})",
             live_dtype=run["dtype"], live_threads=str(run["torch_threads"]),
             live_cv_auc=f3(cr["router_cv"]["auc"]), live_cv_apgr=f3(cr["router_cv"]["apgr"]))
    blocks["live_acc"] = "\n".join(
        ["| source | n | small | big |", "|---|---|---|---|"]
        + [f"| {s} | {x['n']} | {pc(x['small'], 1)} | {pc(x['big'], 1)} |" for s, x in sorted(bs.items())]
        + [f"| **all** | {n} | {pc(acc['small'], 1)} | {pc(acc['big'], 1)} |",
           f"| oracle (best of the two per question) | {n} | {pc(acc['oracle'], 1)} | |"])
    names = [("router_cv", "trained on the local labels, 5-fold CV"), ("source_prior", "benchmark identity only"),
             ("routerbench_transfer", "RouterBench router, transferred")]
    blocks["live_router"] = "\n".join(
        ["| router for this pair | AUC | APGR | advantage vs random at 30% [95% CI] |", "|---|---|---|---|"]
        + [f"| {lab} | {f3(live[k]['auc'])} | {f3(live[k]['apgr'])} | {ci(live[k]['advantage_vs_random']['30%'])} |"
           for k, lab in names])
    # pricing
    models = pr["models"]
    small, big = "gpt-6-luna", "gpt-6.1-sol"
    v.update(price_date=str(pr["price_date"]),
             price_small=f"{models[small]['input']:.2f} / {models[small]['output']:.2f}",
             price_big=f"{models[big]['input']:.2f} / {models[big]['output']:.2f}", price_small_id=small, price_big_id=big)
    return v, blocks


def render(text: str, v: dict[str, str], blocks: dict[str, str]) -> str:
    for name, body in blocks.items():
        pat = re.compile(rf"(<!--BEGIN:{name}-->)\n.*?\n?(<!--END:{name}-->)", re.S)
        if not pat.search(text):
            raise SystemExit(f"README.md has no block {name}")
        text = pat.sub(lambda m, body=body: m.group(1) + "\n" + body + "\n" + m.group(2), text)
    used = set()

    def sub(m: re.Match) -> str:
        used.add(m.group(1))
        if m.group(1) not in v:
            raise SystemExit(f"README.md uses unknown value {m.group(1)}")
        return f"<!--v:{m.group(1)}-->{v[m.group(1)]}<!--/v-->"

    text = re.sub(r"<!--v:(\w+)-->.*?<!--/v-->", sub, text)
    return text


def check_exports(D: dict) -> list[str]:
    """web/data and docs/data must hold exactly what the results files say."""
    bad = []
    off = D["off"][PAIRS[0]]
    for base in ("web", "docs"):
        d = ROOT / base / "data"
        o = json.loads((d / "offline.json").read_text(encoding="utf-8"))
        for key in ("test", "test_curve", "test_oracle", "cv", "leave_family_out", "n_test", "selected"):
            if o.get(key) != json.loads(json.dumps(off[key])):
                bad.append(f"{base}/data/offline.json differs from results for '{key}'")
        e = json.loads((d / "eval.json").read_text(encoding="utf-8"))
        p = D["pred"][PAIRS[0]]
        for k, col in (("p", "p_hard"), ("sq", "s_score"), ("bq", "b_score")):
            if not np.allclose(e[k], p[col].to_numpy(), atol=1e-5):
                bad.append(f"{base}/data/eval.json '{k}' differs from the test predictions")
        for k, col in (("sc", "s_cost"), ("bc", "b_cost")):
            if not np.allclose(e[k], p[col].to_numpy(), atol=1e-7):
                bad.append(f"{base}/data/eval.json '{k}' differs from the test predictions")
        if e["n"] != off["n_test"]:
            bad.append(f"{base}/data/eval.json n != n_test")
        if json.loads((d / "live.json").read_text(encoding="utf-8")) != D["live"]:
            bad.append(f"{base}/data/live.json differs from results/live.json")
    return bad


def check_pricing(D: dict) -> list[str]:
    bad = []
    pr = D["pricing"]
    gw = yaml.safe_load((ROOT / "config" / "gateway.yaml").read_text(encoding="utf-8"))
    for name, t in gw["tiers"].items():
        pa = t.get("price_as")
        if pa and pa not in pr["models"]:
            bad.append(f"gateway tier {name} prices as {pa!r}, which is not in pricing.yaml")
    ui = (ROOT / "web" / "app.js").read_text(encoding="utf-8") + (ROOT / "web" / "index.html").read_text(encoding="utf-8")
    if re.search(r"price_date|2026-11-12", ui):
        bad.append("the UI hard-codes a price date")
    return bad


def main() -> int:
    D = load()
    v, blocks = values(D)
    readme = ROOT / "README.md"
    old = readme.read_text(encoding="utf-8")
    new = render(old, v, blocks)
    problems = check_exports(D) + check_pricing(D)
    if "--write" in sys.argv:
        readme.write_text(new, encoding="utf-8")
        print("README.md updated")
    elif new != old:
        problems.append("README.md is out of sync with results/ (run: python scripts/check_numbers.py --write)")
    for p in problems:
        print("MISMATCH:", p)
    if "--check" in sys.argv and not problems:
        print("numbers in sync")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
