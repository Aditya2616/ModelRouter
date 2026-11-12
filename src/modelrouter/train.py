"""Train and compare difficulty classifiers on RouterBench, write results/offline.json.

Protocol
--------
* 85% development / 15% held-out test, stratified by sub-benchmark, split by prompt.
* 4-fold cross-validation on the development part gives out-of-fold scores for every candidate
  (calibration happens inside each fold, so the scores are honest).
* The candidate with the best out-of-fold APGR is refitted on the development part and scored once on
  the test part.
* Leave-one-benchmark-family-out: train on the other families, test on the held-out family. This
  shows how much of the in-distribution skill is "which benchmark is this" rather than difficulty.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from . import evaluate as ev
from .features import LEXICAL_NAMES, featurize

N_EMB = 384
SEED = 0
TARGETS = {
    "gain": lambda o: (o.b_score > o.s_score).astype(int),        # the big model does strictly better
    "small_fail": lambda o: (o.s_score < 0.5).astype(int),        # the small model gets it wrong
}


def make_models(kind: str):
    """Each returns a calibrated classifier. `lex` models see only the 16 lexical features."""
    if kind == "lr_lex":
        base = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        return CalibratedClassifierCV(base, method="sigmoid", cv=3), slice(N_EMB, None)
    if kind == "lr":
        base = make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        return CalibratedClassifierCV(base, method="sigmoid", cv=3), slice(None)
    if kind == "hgb":
        base = HistGradientBoostingClassifier(max_iter=250, learning_rate=0.06, max_leaf_nodes=24,
                                              l2_regularization=1.0, random_state=SEED)
        return CalibratedClassifierCV(base, method="isotonic", cv=3), slice(None)
    if kind == "rf":
        base = RandomForestClassifier(n_estimators=200, min_samples_leaf=4, max_features="sqrt",
                                      n_jobs=3, random_state=SEED)
        return CalibratedClassifierCV(base, method="isotonic", cv=3), slice(None)
    if kind == "mlp":
        base = make_pipeline(StandardScaler(), MLPClassifier((128,), alpha=1e-2, early_stopping=True,
                                                             max_iter=200, random_state=SEED))
        return CalibratedClassifierCV(base, method="sigmoid", cv=3), slice(None)
    raise ValueError(kind)


KINDS = ["lr_lex", "lr", "hgb", "rf", "mlp"]


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(tot)


def fit_predict(kind: str, Xtr, ytr, Xte) -> np.ndarray:
    model, cols = make_models(kind)
    model.fit(Xtr[:, cols], ytr)
    return model.predict_proba(Xte[:, cols])[:, 1]


def family_prior(train_fam, ytr, test_fam) -> np.ndarray:
    """Router that only knows the benchmark family: P(target) = family mean. A strong baseline if the
    benchmark identity explains most of the signal."""
    s = pd.Series(ytr).groupby(np.asarray(train_fam)).mean()
    return pd.Series(test_fam).map(s).fillna(float(np.mean(ytr))).to_numpy(dtype=float)


def metrics_block(y, p, o: ev.Outcomes) -> dict:
    out = {"auc": float(roc_auc_score(y, p)), "brier": float(brier_score_loss(y, p)), "ece": ece(y, p)}
    out.update(ev.summarize(p, o))
    return out


def run(pair: str = "gpt35_vs_gpt4", kinds=None, targets=("gain", "small_fail"),
        out_dir: Path = Path("results"), model_dir: Path = Path("models"), log=print) -> dict:
    kinds = kinds or KINDS
    df = D.load_slim()
    X = featurize(df.prompt.tolist(), cache=Path("data/emb_routerbench.npy"))
    arr = D.pair_arrays(df, pair)
    o_all = ev.Outcomes(**arr)
    fam = df["family"].to_numpy()
    task = df["task"].to_numpy()

    strat = pd.Series(task).where(pd.Series(task).map(pd.Series(task).value_counts()) >= 4, "rare")
    idx = np.arange(len(df))
    dev, test = train_test_split(idx, test_size=0.15, random_state=SEED, stratify=strat)
    log(f"[{pair}] dev={len(dev)} test={len(test)}")

    result: dict = {"pair": pair, "small": D.PAIRS[pair][0], "big": D.PAIRS[pair][1],
                    "n_total": len(df), "n_dev": len(dev), "n_test": len(test),
                    "recorded_price_note": "costs are RouterBench's recorded per-call costs at late-2023 prices",
                    "cv": {}, "families": df["family"].value_counts().to_dict()}
    skf = StratifiedKFold(4, shuffle=True, random_state=SEED)
    o_dev = o_all.take(dev)
    oof_store: dict = {}
    for tname in targets:
        y_dev = TARGETS[tname](o_dev)
        for kind in kinds:
            t0 = time.time()
            oof = np.zeros(len(dev))
            for tr, va in skf.split(dev, strat.to_numpy()[dev]):
                oof[va] = fit_predict(kind, X[dev][tr], y_dev[tr], X[dev][va])
            m = metrics_block(y_dev, oof, o_dev)
            m["seconds"] = round(time.time() - t0, 1)
            result["cv"][f"{tname}/{kind}"] = m
            oof_store[(tname, kind)] = oof
            log(f"  cv {tname:10s} {kind:7s} auc={m['auc']:.3f} brier={m['brier']:.3f} ece={m['ece']:.3f} "
                f"apgr={m['apgr']:.3f} aiq={m['aiq']:.4f} cpt95={m['cpt95']:.3f} ({m['seconds']}s)")
    # family-prior baseline (benchmark identity only), same folds
    for tname in targets:
        y_dev = TARGETS[tname](o_dev)
        oof = np.zeros(len(dev))
        for tr, va in skf.split(dev, strat.to_numpy()[dev]):
            oof[va] = family_prior(fam[dev][tr], y_dev[tr], fam[dev][va])
        result["cv"][f"{tname}/family_prior"] = metrics_block(y_dev, oof, o_dev)
        result["cv"][f"{tname}/task_prior"] = metrics_block(
            y_dev, _task_prior_oof(task[dev], y_dev, skf, strat.to_numpy()[dev]), o_dev)

    best = max((k for k in result["cv"] if k.split("/")[1] in kinds), key=lambda k: result["cv"][k]["apgr"])
    tname, kind = best.split("/")
    result["selected"] = {"target": tname, "model": kind, "rule": "best out-of-fold APGR on development data"}
    log(f"selected {best}")

    # held-out test, evaluated once
    y_dev = TARGETS[tname](o_dev)
    model, cols = make_models(kind)
    model.fit(X[dev][:, cols], y_dev)
    p_test = model.predict_proba(X[test][:, cols])[:, 1]
    o_test = o_all.take(test)
    y_test = TARGETS[tname](o_test)
    result["test"] = metrics_block(y_test, p_test, o_test)
    cv_test = ev.curve(p_test, o_test)
    grid = np.linspace(0, 1, 101)
    lo, hi = ev.random_band(o_test, grid)
    oracle = ev.curve((o_test.b_score - o_test.s_score) + 1e-9 * np.random.default_rng(0).random(len(test)), o_test)
    result["test_advantage_vs_random"] = ev.paired_bootstrap_advantage(p_test, o_test)
    result["test_curve"] = ev.downsample(cv_test, 300)
    result["test_random_band"] = {"frac": grid.tolist(), "lo": np.round(lo, 6).tolist(), "hi": np.round(hi, 6).tolist()}
    result["test_oracle_curve"] = ev.downsample(oracle, 300)
    result["test_oracle"] = {"cpt95": ev.cpt(oracle, o_test, 0.95), "apgr": ev.apgr(oracle, o_test)}
    fp = family_prior(fam[dev], y_dev, fam[test])
    result["test_family_prior"] = metrics_block(y_test, fp, o_test)
    result["test_by_family"] = _by_family(p_test, o_test, y_test, fam[test])

    # leave-one-family-out
    lofo = {}
    for f in sorted(set(fam)):
        te = np.flatnonzero(fam == f)
        tr = np.flatnonzero(fam != f)
        if len(te) < 50:
            continue
        o_tr, o_te = o_all.take(tr), o_all.take(te)
        ytr, yte = TARGETS[tname](o_tr), TARGETS[tname](o_te)
        p = fit_predict(kind, X[tr], ytr, X[te])
        row = {"n": len(te), "small_quality": float(o_te.s_score.mean()), "big_quality": float(o_te.b_score.mean())}
        if 0 < yte.mean() < 1:
            row["auc"] = float(roc_auc_score(yte, p))
        row["apgr"] = ev.apgr(ev.curve(p, o_te), o_te)
        adv = ev.paired_bootstrap_advantage(p, o_te, fracs=(0.3,), n_boot=200)
        row["advantage_vs_random_30pct"] = adv["30%"]
        lofo[f] = row
        log(f"  lofo {f:14s} n={len(te):5d} auc={row.get('auc', float('nan')):.3f} apgr={row['apgr']:.3f} "
            f"adv30={adv['30%']['mean']:+.4f} [{adv['30%']['lo']:+.4f},{adv['30%']['hi']:+.4f}]")
    result["leave_family_out"] = lofo

    # final model: refit on development data, calibrated; save the bundle and the scored test set
    model_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    p_dev_oof = oof_store[(tname, kind)]
    bundle = {"model": model, "cols": cols, "target": tname, "kind": kind, "pair": pair,
              "n_emb": N_EMB, "lexical": LEXICAL_NAMES,
              "dev_score_quantiles": np.quantile(p_dev_oof, np.linspace(0, 1, 101)).tolist()}
    joblib.dump(bundle, model_dir / f"router_{pair}.joblib", compress=3)
    pred = pd.DataFrame({
        "sample_id": df["sample_id"].to_numpy()[test], "task": task[test], "family": fam[test],
        "p_hard": np.round(p_test, 5), "s_score": o_test.s_score, "b_score": o_test.b_score,
        "s_cost": o_test.s_cost, "b_cost": o_test.b_cost, "y": y_test})
    pred.to_parquet(out_dir / f"test_predictions_{pair}.parquet", index=False)
    (out_dir / f"offline_{pair}.json").write_text(json.dumps(result, indent=1, default=float))
    return result


def _task_prior_oof(task_dev, y_dev, skf, strat_dev):
    oof = np.zeros(len(y_dev))
    for tr, va in skf.split(np.arange(len(y_dev)), strat_dev):
        oof[va] = family_prior(task_dev[tr], y_dev[tr], task_dev[va])
    return oof


def _by_family(p, o, y, fam) -> dict:
    out = {}
    for f in sorted(set(fam)):
        m = fam == f
        if m.sum() < 30:
            continue
        row = {"n": int(m.sum()), "small_quality": float(o.s_score[m].mean()),
               "big_quality": float(o.b_score[m].mean()), "target_rate": float(y[m].mean())}
        if 0 < y[m].mean() < 1:
            row["within_family_auc"] = float(roc_auc_score(y[m], p[m]))
        out[f] = row
    return out
