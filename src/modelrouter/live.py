"""Small live experiment on this machine: a local small/big pair (Qwen2.5-0.5B and 1.5B Instruct).

240 multiple-choice questions with known answers (ARC-Easy, ARC-Challenge, CommonsenseQA, MMLU; 60 each)
are answered by both models through the same provider code the gateway uses. Their real correctness
becomes the training signal for a router for this pair. This is a CPU-scale demonstration of the
mechanism; the offline RouterBench analysis is the main evidence.
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .features import LEXICAL_NAMES, featurize
from .providers import LocalProvider

SMALL, BIG = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"
SOURCES = {
    "arc-easy": ("allenai/ai2_arc", "ARC-Easy/test-00000-of-00001.parquet"),
    "arc-challenge": ("allenai/ai2_arc", "ARC-Challenge/test-00000-of-00001.parquet"),
    "commonsenseqa": ("tau/commonsense_qa", "data/validation-00000-of-00001.parquet"),
    "mmlu": ("cais/mmlu", "all/test-00000-of-00001.parquet"),
}
N_PER = 60
LETTERS = "ABCDE"
PREFIX = "The correct answer is"   # forced start of the reply; the models then emit the letter
DATA = Path("data/live")


def _download(repo: str, path: str) -> pd.DataFrame:
    DATA.mkdir(parents=True, exist_ok=True)
    dest = DATA / (repo.replace("/", "__") + "__" + path.replace("/", "__"))
    if not dest.exists():
        urllib.request.urlretrieve(HF.format(repo=repo, path=path), dest)
    return pd.read_parquet(dest)


def _format(q: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"Answer the multiple-choice question with the letter of the correct option only.\n\n{q}\n{opts}\nAnswer:"


def build_items(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for src, (repo, path) in SOURCES.items():
        df = _download(repo, path)
        pick = rng.permutation(len(df))[: N_PER * 3]
        got = 0
        for i in pick:
            r = df.iloc[int(i)]
            if src == "mmlu":
                options, gold = list(r["choices"]), LETTERS[int(r["answer"])]
            else:
                ch = r["choices"]
                options, labels = list(ch["text"]), list(ch["label"])
                key = str(r["answerKey"])
                if key not in labels or len(options) > 5:
                    continue
                gold = LETTERS[labels.index(key)]
            if len(options) > 5:
                continue
            rows.append({"source": src, "prompt": _format(str(r["question"]).strip(), [str(o) for o in options]),
                         "gold": gold})
            got += 1
            if got == N_PER:
                break
    return pd.DataFrame(rows)


_LETTER = re.compile(r"\b([A-E])\b")


def parse_letter(text: str) -> str | None:
    m = _LETTER.search(text.strip().upper()[:12]) or _LETTER.search(text.upper())
    return m.group(1) if m else None


def run_models(items: pd.DataFrame, out: Path = DATA / "runs.parquet", threads: int = 3, log=print) -> pd.DataFrame:
    """Answer every item with both models, one model in memory at a time (the machine is memory-bound).
    Resumable per model: results are checkpointed to `out`. Records answer, correctness, latency, logprob."""
    import gc

    runs = pd.read_parquet(out) if out.exists() else pd.DataFrame({"i": range(len(items)), "source": items.source.values,
                                                                      "gold": items.gold.values})
    for tag, model in (("s", SMALL), ("b", BIG)):
        if f"{tag}_ok" in runs.columns and runs[f"{tag}_ok"].notna().all():
            continue
        prov = LocalProvider(threads=threads)
        recs = []
        t0 = time.time()
        for i in range(len(items)):
            it = items.iloc[i]
            c = prov.complete(model, [{"role": "user", "content": it.prompt}], max_tokens=4, temperature=0.0, prefix=PREFIX)
            recs.append({f"{tag}_text": c.text, f"{tag}_ok": float(parse_letter(c.text) == it.gold),
                         f"{tag}_latency": c.latency_s,
                         f"{tag}_logprob": c.mean_logprob if c.mean_logprob is not None else np.nan,
                         f"{tag}_tokens": c.prompt_tokens})
            if (i + 1) % 20 == 0:
                log(f"  {model.split('/')[1]}: {i + 1}/{len(items)} items, {time.time() - t0:.0f}s")
        runs = pd.concat([runs, pd.DataFrame(recs)], axis=1)
        runs.to_parquet(out, index=False)
        del prov
        gc.collect()
    return runs


def _router_model():
    base = make_pipeline(StandardScaler(), LogisticRegression(C=0.03, max_iter=3000))
    return CalibratedClassifierCV(base, method="sigmoid", cv=3)


def analyse(items: pd.DataFrame, runs: pd.DataFrame, out_dir: Path = Path("results"), model_dir: Path = Path("models"),
            log=print) -> dict:
    X = featurize(items.prompt.tolist(), cache=DATA / "emb_live.npy")
    o = ev.Outcomes(runs.s_ok.to_numpy(float), runs.b_ok.to_numpy(float),
                    # cost proxy: parameters x prompt tokens (billions of parameter-tokens); see README
                    runs.s_tokens.to_numpy(float) * 0.494, runs.b_tokens.to_numpy(float) * 1.544)
    y = (o.b_score > o.s_score).astype(int)
    src = items.source.to_numpy()
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    strat = pd.Series(src).astype(str) + "_" + pd.Series(y).astype(str)
    oof = np.zeros(len(y))
    for tr, va in skf.split(X, strat):
        m = _router_model()
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
    # benchmark-identity baseline (what the source alone predicts)
    oof_fam = np.zeros(len(y))
    for tr, va in skf.split(X, strat):
        oof_fam[va] = pd.Series(src[va]).map(pd.Series(y[tr]).groupby(src[tr]).mean()).to_numpy(float)
    # RouterBench-trained router transferred to this pair
    transfer = None
    rb = Path("models/router_gpt35_vs_gpt4.joblib")
    if rb.exists():
        from .router import DifficultyRouter

        transfer = DifficultyRouter.load(rb).model.predict_proba(X[:, DifficultyRouter.load(rb).cols])[:, 1]

    res: dict = {
        "small": SMALL, "big": BIG, "n": len(items), "per_source": N_PER,
        "accuracy": {"small": float(o.s_score.mean()), "big": float(o.b_score.mean()),
                     "oracle": float(np.maximum(o.s_score, o.b_score).mean())},
        "share_big_strictly_better": float(y.mean()),
        "mean_latency_s": {"small": float(runs.s_latency.mean()), "big": float(runs.b_latency.mean())},
        "by_source": {s: {"n": int((src == s).sum()), "small": float(o.s_score[src == s].mean()),
                          "big": float(o.b_score[src == s].mean())} for s in sorted(set(src))},
        "cost_proxy_note": "cost proxy = prompt tokens x model parameters (billions); a compute proxy, not dollars",
    }
    cands = {"router_cv": oof, "source_prior": oof_fam}
    if transfer is not None:
        cands["routerbench_transfer"] = transfer
    for name, p in cands.items():
        cv = ev.curve(p, o)
        entry = ev.summarize(p, o)
        entry["auc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
        entry["advantage_vs_random"] = ev.paired_bootstrap_advantage(p, o, fracs=(0.3, 0.5), n_boot=400)
        entry["curve"] = ev.downsample(cv, 200)
        res[name] = entry
        log(f"  {name:22s} auc={entry['auc']:.3f} apgr={entry['apgr']:.3f} cpt95={entry['cpt95']:.3f} "
            f"adv@30%={entry['advantage_vs_random']['30%']['mean']:+.4f} "
            f"[{entry['advantage_vs_random']['30%']['lo']:+.4f},{entry['advantage_vs_random']['30%']['hi']:+.4f}]")
    grid = np.linspace(0, 1, 101)
    lo, hi = ev.random_band(o, grid)
    res["random_band"] = {"frac": grid.tolist(), "lo": np.round(lo, 6).tolist(), "hi": np.round(hi, 6).tolist()}
    # within-source AUC of the cross-validated router
    res["within_source_auc"] = {s: float(roc_auc_score(y[src == s], oof[src == s]))
                                for s in sorted(set(src)) if 0 < y[src == s].mean() < 1}
    # final router for the local gateway: fit on all items
    final = _router_model()
    final.fit(X, y)
    bundle = {"model": final, "cols": slice(None), "target": "gain", "kind": "lr_local", "pair": "local_qwen",
              "n_emb": 384, "lexical": LEXICAL_NAMES,
              "dev_score_quantiles": np.quantile(oof, np.linspace(0, 1, 101)).tolist()}
    model_dir.mkdir(exist_ok=True)
    joblib.dump(bundle, model_dir / "router_local_qwen.joblib", compress=3)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "live.json").write_text(json.dumps(res, indent=1, default=float))
    pd.DataFrame({"source": src, "p_hard": np.round(oof, 5), "s_ok": o.s_score, "b_ok": o.b_score,
                  "s_latency": runs.s_latency.round(4), "b_latency": runs.b_latency.round(4)}).to_parquet(
        out_dir / "live_items.parquet", index=False)
    return res


def run(log=print) -> dict:
    items = build_items()
    log(f"{len(items)} items: {items.source.value_counts().to_dict()}")
    runs = run_models(items, log=log)
    return analyse(items, runs, log=log)
