"""Live experiment on this machine: a local small/big pair (Qwen2.5-0.5B and 1.5B Instruct).

About 3,000 multiple-choice questions with known answers (ARC-Easy, ARC-Challenge, CommonsenseQA, MMLU; 750 each) are
answered by both models with the same prompt and the same forced answer prefix. Their real correctness is the
training signal for a router for this pair. The items are split by source and seed into train / validation / test
with no prompt on both sides, routers are fitted on train, chosen on validation and scored once on test.
The offline RouterBench analysis is the main evidence; this is the end-to-end check with two real models.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import evaluate as ev
from .features import LEXICAL_NAMES, featurize
from .providers import LocalProvider

SMALL, BIG = "Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"
PARAMS_B = {"s": 0.494, "b": 1.544}   # billions of parameters (from the model cards), for the compute cost proxy
# (dataset repo, parquet files that carry gold labels, licence as stated on the dataset card)
SOURCES = {
    "arc-easy": ("allenai/ai2_arc", ["ARC-Easy/train-00000-of-00001.parquet", "ARC-Easy/validation-00000-of-00001.parquet",
                                     "ARC-Easy/test-00000-of-00001.parquet"], "CC-BY-SA-4.0"),
    "arc-challenge": ("allenai/ai2_arc", ["ARC-Challenge/train-00000-of-00001.parquet",
                                          "ARC-Challenge/validation-00000-of-00001.parquet",
                                          "ARC-Challenge/test-00000-of-00001.parquet"], "CC-BY-SA-4.0"),
    "commonsenseqa": ("tau/commonsense_qa", ["data/train-00000-of-00001.parquet", "data/validation-00000-of-00001.parquet"],
                      "MIT"),
    "mmlu": ("cais/mmlu", ["all/test-00000-of-00001.parquet"], "MIT"),
}
N_PER = 750
SPLIT_PER = {"train": 500, "val": 125, "test": 125}   # per source: 2,000 / 500 / 500 in total
ITEM_SEED, SPLIT_SEED = 0, 1
LETTERS = "ABCDE"
PREFIX = "The correct answer is"   # forced start of the reply; the models then emit the letter
BATCH = 32
LAT_N, LAT_WARMUP = 120, 5          # single-request latency: items timed per tier (from the test split), warm-up calls
DATA = Path("data/live")


def _fetch(repo: str, path: str) -> pd.DataFrame:
    from huggingface_hub import hf_hub_download

    return pd.read_parquet(hf_hub_download(repo, path, repo_type="dataset"))


def _format(q: str, options: list[str]) -> str:
    opts = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
    return f"Answer the multiple-choice question with the letter of the correct option only.\n\n{q}\n{opts}\nAnswer:"


def _key(q: str, options: list[str]) -> str:
    return " ".join((q + " " + " ".join(options)).lower().split())


def _rows(src: str) -> list[dict]:
    repo, paths, _ = SOURCES[src]
    out = []
    for path in paths:
        df = _fetch(repo, path)
        for r in df.itertuples(index=False):
            if src == "mmlu":
                options, gold = [str(o) for o in r.choices], LETTERS[int(r.answer)]
            else:
                options, labels, key = [str(o) for o in r.choices["text"]], list(r.choices["label"]), str(r.answerKey)
                if key not in labels:
                    continue
                gold = LETTERS[labels.index(key)]
            if len(options) > 5 or len(options) < 2:
                continue
            q = str(r.question).strip()
            out.append({"source": src, "prompt": _format(q, options), "gold": gold, "key": _key(q, options)})
    return out


def build_items(cache: Path = DATA / "items.parquet") -> pd.DataFrame:
    """About 3,000 items: 750 per source, de-duplicated across the whole pool by normalised question and options,
    then split per source into train / validation / test by seed. Deterministic."""
    if cache.exists():
        return pd.read_parquet(cache)
    rng = np.random.default_rng(ITEM_SEED)
    seen: set[str] = set()
    parts = []
    for src in SOURCES:
        pool = _rows(src)
        picked = []
        for i in rng.permutation(len(pool)):
            r = pool[int(i)]
            if r["key"] in seen:
                continue
            seen.add(r["key"])
            picked.append(r)
            if len(picked) == N_PER:
                break
        assert len(picked) == N_PER, f"{src}: only {len(picked)} unique items"
        sp = np.random.default_rng(SPLIT_SEED).permutation(N_PER)
        split = np.empty(N_PER, dtype=object)
        a, b = SPLIT_PER["train"], SPLIT_PER["train"] + SPLIT_PER["val"]
        split[sp[:a]], split[sp[a:b]], split[sp[b:]] = "train", "val", "test"
        for r, s in zip(picked, split, strict=True):
            parts.append({**r, "split": s})
    items = pd.DataFrame(parts)
    assert items.key.is_unique
    items = items.drop(columns="key")
    cache.parent.mkdir(parents=True, exist_ok=True)
    items.to_parquet(cache, index=False)
    return items


_LETTER = re.compile(r"\b([A-E])\b")


def parse_letter(text: str) -> str | None:
    m = _LETTER.search(text.strip().upper()[:12]) or _LETTER.search(text.upper())
    return m.group(1) if m else None


def _gpu_state() -> dict | None:
    """GPU utilisation and memory in use right before a timing run (the machine is shared); None without nvidia-smi."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
        u, m = (int(x) for x in r.stdout.strip().splitlines()[0].split(","))
        return {"gpu_util_pct_before": u, "gpu_mem_used_mib_before": m}
    except Exception:  # noqa: BLE001
        return None


def run_models(items: pd.DataFrame, out: Path = DATA / "runs.parquet", meta_path: Path = DATA / "meta.json",
               threads: int = 2, device: str | None = None, log=print) -> tuple[pd.DataFrame, dict]:
    """Answer every item with both models, one model in memory at a time, in batches, with the same harness.
    Per model, after the batched pass, times single requests (batch size 1) on a subset of the test split.
    Resumable per model: results are checkpointed to `out` and `meta_path`."""
    runs = pd.read_parquet(out) if out.exists() else pd.DataFrame({"source": items.source.values, "split": items.split.values,
                                                                      "gold": items.gold.values})
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    prov = LocalProvider(threads=threads, device=device)
    meta.update(device=prov.device, dtype=prov.dtype_name, torch_threads=threads, batch_size=BATCH)
    if prov.device.startswith("cuda"):
        import torch

        meta["gpu"] = torch.cuda.get_device_name(0)
    lat_idx = np.flatnonzero(items.split.to_numpy() == "test")[:LAT_N]
    for tag, model in (("s", SMALL), ("b", BIG)):
        if f"{tag}_ok" in runs.columns:
            continue
        prompts = items.prompt.tolist()
        prov.complete_batch(model, prompts[:BATCH], max_tokens=4, prefix=PREFIX, batch_size=BATCH)  # warm-up (load, kernels)
        t0 = time.perf_counter()
        comps = prov.complete_batch(model, prompts, max_tokens=4, prefix=PREFIX, batch_size=BATCH)
        wall = time.perf_counter() - t0
        log(f"  {model.split('/')[1]}: {len(prompts)} items in {wall:.0f}s (batches of {BATCH})")
        gold = items.gold.tolist()
        runs[f"{tag}_text"] = [c.text for c in comps]
        runs[f"{tag}_ok"] = [float(parse_letter(c.text) == g) for c, g in zip(comps, gold, strict=True)]
        runs[f"{tag}_tokens"] = [c.prompt_tokens for c in comps]
        runs[f"{tag}_logprob"] = [c.mean_logprob if c.mean_logprob is not None else np.nan for c in comps]
        tok_total = int(sum(c.prompt_tokens for c in comps))
        # single-request latency: one prompt at a time, as the gateway sends them, after warm-up calls
        for i in lat_idx[:LAT_WARMUP]:
            prov.complete(model, [{"role": "user", "content": prompts[int(i)]}], max_tokens=4, prefix=PREFIX)
        state = _gpu_state()
        lats, toks = [], []
        for i in lat_idx:
            c = prov.complete(model, [{"role": "user", "content": prompts[int(i)]}], max_tokens=4, temperature=0.0, prefix=PREFIX)
            lats.append(c.latency_s)
            toks.append(c.prompt_tokens)
        meta[tag] = {
            "batched": {"n": len(prompts), "wall_s": wall, "items_per_s": len(prompts) / wall,
                        "prompt_tokens": tok_total, "prompt_tokens_per_s": tok_total / wall},
            "single": {"n": len(lats), "mean_s": float(np.mean(lats)), "median_s": float(np.median(lats)),
                       "p95_s": float(np.percentile(lats, 95)), "mean_prompt_tokens": float(np.mean(toks)),
                       "prompt_tokens_per_s": float(sum(toks) / sum(lats)), "state_before": state},
        }
        log(f"  {model.split('/')[1]}: single-request mean {np.mean(lats) * 1000:.0f} ms over {len(lats)} items")
        runs.to_parquet(out, index=False)
        meta_path.write_text(json.dumps(meta, indent=1))
        prov.release()
    return runs, meta


# ---------------------------------------------------------------------------------------------------- routers
def _lr(C: float):
    return CalibratedClassifierCV(make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)), method="sigmoid", cv=3)


def _hgb():
    return CalibratedClassifierCV(HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150,
                                                                 random_state=0), method="sigmoid", cv=3)


CANDIDATES = {**{f"lr_C{c}": (lambda c=c: _lr(c)) for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)}, "hgb": _hgb}


def _scores(p: np.ndarray, o: ev.Outcomes, y: np.ndarray) -> dict:
    s = ev.summarize(p, o)
    s["auc"] = float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else None
    return s


def _bootstrap(p: np.ndarray, o: ev.Outcomes, y: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict:
    """95% bootstrap intervals over test items for AUC, APGR and CPT 95%."""
    rng = np.random.default_rng(seed)
    n = len(p)
    auc, apg, cpt95 = [], [], []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if not 0 < y[i].mean() < 1:
            continue
        ob = o.take(i)
        cv = ev.curve(p[i], ob)
        auc.append(roc_auc_score(y[i], p[i]))
        apg.append(ev.apgr(cv, ob))
        cpt95.append(ev.cpt(cv, ob, 0.95))
    iv = lambda v: {"lo": float(np.nanpercentile(v, 2.5)), "hi": float(np.nanpercentile(v, 97.5))}  # noqa: E731
    return {"auc": iv(auc), "apgr": iv(apg), "cpt95": iv(cpt95), "n_boot": n_boot}


def _evaluate(p: np.ndarray, o: ev.Outcomes, y: np.ndarray, src: np.ndarray) -> dict:
    e = _scores(p, o, y)
    e["interval"] = _bootstrap(p, o, y)
    e["advantage_vs_random"] = ev.paired_bootstrap_advantage(p, o, fracs=(0.3, 0.5), n_boot=1000)
    e["curve"] = ev.downsample(ev.curve(p, o), 200)
    e["within_source_auc"] = {s: float(roc_auc_score(y[src == s], p[src == s])) for s in sorted(set(src))
                              if 0 < y[src == s].mean() < 1}
    return e


def _source_prior(src_tr: np.ndarray, y_tr: np.ndarray, src: np.ndarray) -> np.ndarray:
    prior = pd.Series(y_tr).groupby(src_tr).mean()
    return pd.Series(src).map(prior).to_numpy(float)


def _transfer(X: np.ndarray) -> np.ndarray | None:
    rb = Path("models/router_gpt35_vs_gpt4.joblib")
    if not rb.exists():
        return None
    from .router import DifficultyRouter

    r = DifficultyRouter.load(rb)
    return r.model.predict_proba(X[:, r.cols])[:, 1]


def analyse(items: pd.DataFrame, runs: pd.DataFrame, meta: dict, out_dir: Path = Path("results"),
            model_dir: Path = Path("models"), log=print) -> dict:
    h = hashlib.sha256("\n".join(items.prompt).encode()).hexdigest()[:10]
    X = featurize(items.prompt.tolist(), cache=DATA / f"emb_live_{h}.npy")
    sp = items.split.to_numpy()
    src = items.source.to_numpy()
    s_ok, b_ok = runs.s_ok.to_numpy(float), runs.b_ok.to_numpy(float)
    o_all = ev.Outcomes(s_ok, b_ok, runs.s_tokens.to_numpy(float) * PARAMS_B["s"], runs.b_tokens.to_numpy(float) * PARAMS_B["b"])
    y = (b_ok > s_ok).astype(int)
    tr, va, te = (np.flatnonzero(sp == k) for k in ("train", "val", "test"))
    assert len(tr) + len(va) + len(te) == len(items) and not (set(tr) & set(va) | set(tr) & set(te) | set(va) & set(te))
    assert not (set(items.prompt[tr]) & set(items.prompt[te])) and not (set(items.prompt[tr]) & set(items.prompt[va]))
    assert not (set(items.prompt[va]) & set(items.prompt[te]))
    ob_split = {k: o_all.take(idx) for k, idx in (("train", tr), ("val", va), ("test", te))}

    # 1. local router: pick the candidate with the best validation APGR (train-only fits)
    val_apgr, fitted = {}, {}
    for name, make in CANDIDATES.items():
        m = make()
        m.fit(X[tr], y[tr])
        fitted[name] = m
        val_apgr[name] = ev.apgr(ev.curve(m.predict_proba(X[va])[:, 1], ob_split["val"]), ob_split["val"])
        log(f"  local candidate {name:10s} val APGR {val_apgr[name]:.3f}")
    best = max(val_apgr, key=val_apgr.get)
    cands: dict[str, dict[str, np.ndarray]] = {
        "local": {"val": fitted[best].predict_proba(X[va])[:, 1], "test": fitted[best].predict_proba(X[te])[:, 1]},
        "source_prior": {k: _source_prior(src[tr], y[tr], src[idx]) for k, idx in (("val", va), ("test", te))},
    }
    t_all = _transfer(X)
    if t_all is not None:
        cands["routerbench_transfer"] = {"val": t_all[va], "test": t_all[te]}

    res: dict = {
        "small": SMALL, "big": BIG, "n": len(items), "per_source": N_PER, "seeds": {"items": ITEM_SEED, "split": SPLIT_SEED},
        "licences": {s: SOURCES[s][2] for s in SOURCES},
        "split": {k: {"n": int(len(idx)), "per_source": {s: int((src[idx] == s).sum()) for s in sorted(set(src))}}
                  for k, idx in (("train", tr), ("val", va), ("test", te))},
        "run": {k: meta[k] for k in ("device", "dtype", "torch_threads", "batch_size") if k in meta} | (
            {"gpu": meta["gpu"]} if "gpu" in meta else {}),
        "latency": {t: {"batched": meta[k]["batched"], "single": meta[k]["single"]} for t, k in (("small", "s"), ("big", "b"))},
        "accuracy": {"small": float(s_ok.mean()), "big": float(b_ok.mean()), "oracle": float(np.maximum(s_ok, b_ok).mean())},
        "accuracy_test": {"small": float(s_ok[te].mean()), "big": float(b_ok[te].mean()),
                          "oracle": float(np.maximum(s_ok[te], b_ok[te]).mean())},
        "share_big_strictly_better": float(y.mean()), "share_big_strictly_better_test": float(y[te].mean()),
        "mean_prompt_tokens": {"small": float(runs.s_tokens.mean()), "big": float(runs.b_tokens.mean())},
        "by_source": {s: {"n": int((src == s).sum()), "small": float(s_ok[src == s].mean()), "big": float(b_ok[src == s].mean()),
                          "oracle": float(np.maximum(s_ok, b_ok)[src == s].mean())} for s in sorted(set(src))},
        "cost_proxy_note": "cost proxy = prompt tokens x model parameters (billions); a compute proxy, not dollars",
        "local_candidates": {"val_apgr": val_apgr, "chosen": best},
    }
    for name, d in cands.items():
        e = _evaluate(d["test"], ob_split["test"], y[te], src[te])
        e["val"] = _scores(d["val"], ob_split["val"], y[va])
        res[name] = e
        a = e["advantage_vs_random"]["30%"]
        log(f"  {name:22s} test auc={e['auc']:.3f} apgr={e['apgr']:.3f} cpt95={e['cpt95']:.3f} adv@30%={a['mean']:+.4f} "
            f"[{a['lo']:+.4f},{a['hi']:+.4f}]  (val apgr {e['val']['apgr']:.3f})")
    # The gateway uses the deployable router (one that scores a raw prompt, so not the benchmark-identity baseline, which needs
    # the source label) with the best validation APGR; the test numbers are reported, not used for the choice.
    deployable = [k for k in cands if k != "source_prior"]
    chosen = max(deployable, key=lambda k: res[k]["val"]["apgr"])
    res["gateway_router"] = chosen
    res["oracle"] = {"cpt95": ev.cpt(ev.curve(y[te] + 0.0, ob_split["test"]), ob_split["test"], 0.95)}
    grid = np.linspace(0, 1, 101)
    lo, hi = ev.random_band(ob_split["test"], grid)
    res["random_band"] = {"frac": grid.tolist(), "lo": np.round(lo, 6).tolist(), "hi": np.round(hi, 6).tolist()}
    # leave-one-source-out for the chosen local model: fit on the other three sources (train+val), test on the whole held-out one
    trva = np.concatenate([tr, va])
    lofo = {}
    for s in sorted(set(src)):
        a, b = trva[src[trva] != s], np.flatnonzero(src == s)
        m = CANDIDATES[best]()
        m.fit(X[a], y[a])
        p = m.predict_proba(X[b])[:, 1]
        ob = o_all.take(b)
        adv = ev.paired_bootstrap_advantage(p, ob, fracs=(0.3,), n_boot=500)["30%"]
        lofo[s] = {"n": int(len(b)), "auc": float(roc_auc_score(y[b], p)), "apgr": ev.apgr(ev.curve(p, ob), ob),
                   "advantage_vs_random_30pct": adv}
    res["leave_source_out"] = lofo

    # the router the gateway can load: refit on train + validation; score quantiles come from the train-only model on val
    model_dir.mkdir(exist_ok=True)
    final = CANDIDATES[best]()
    final.fit(X[trva], y[trva])
    bundle = {"model": final, "cols": slice(None), "target": "gain", "kind": f"local_{best}", "pair": "local_qwen",
              "n_emb": 384, "lexical": LEXICAL_NAMES,
              "dev_score_quantiles": np.quantile(cands["local"]["val"], np.linspace(0, 1, 101)).tolist()}
    joblib.dump(bundle, model_dir / "router_local_qwen.joblib", compress=3)
    out_dir.mkdir(exist_ok=True)
    (out_dir / "live.json").write_text(json.dumps(res, indent=1, default=float))
    test_rows = {"source": src[te], "p_hard": np.round(cands[chosen]["test"], 5), "s_ok": s_ok[te], "b_ok": b_ok[te]}
    pd.DataFrame(test_rows).to_parquet(out_dir / "live_items.parquet", index=False)
    return res


def run(device: str | None = None, log=print) -> dict:
    items = build_items()
    log(f"{len(items)} items: {items.groupby(['split', 'source']).size().to_dict()}")
    runs, meta = run_models(items, device=device, log=log)
    log(f"device={meta['device']} {meta['dtype']}")
    return analyse(items, runs, meta, log=log)
