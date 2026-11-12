"""Turn the results of a real run into the small JSON files the web UI loads (web/data/*.json).

Only derived numbers are bundled (predicted difficulty, each tier's recorded score and cost, task family).
Prompt text is bundled for a few dozen items from source benchmarks whose own licences clearly allow
redistribution (MMLU: MIT, GSM8K: MIT, HellaSwag: MIT, Winogrande: Apache-2.0), shortened and with
attribution. For every other family the UI shows the label only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from modelrouter import data as D  # noqa: E402

PAIR = "gpt35_vs_gpt4"
REDISTRIBUTABLE = {"mmlu": "MMLU, MIT", "gsm8k": "GSM8K, MIT", "hellaswag": "HellaSwag, MIT", "winogrande": "Winogrande, Apache-2.0"}
OUT = ROOT / "web" / "data"


def snippet(prompt: str, family: str) -> str:
    """Pull the question out of the benchmark prompt template and shorten it."""
    p = prompt.strip()
    if family == "gsm8k":
        # keep only the final question of the few-shot template
        q = p.rsplit("Question:", 1)[-1].split("Answer:")[0].strip()
        return q[:200]
    for cut in ("Please answer with the letter of the correct answer.", "Print only a single choice"):
        p = p.replace(cut, "")
    p = " ".join(p.split())
    return p[:200] + ("..." if len(p) > 200 else "")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    off = json.loads((ROOT / "results" / f"offline_{PAIR}.json").read_text())
    keep = ["pair", "small", "big", "n_total", "n_dev", "n_test", "selected", "cv", "test", "test_curve", "test_random_band",
            "test_oracle_curve", "test_oracle", "test_advantage_vs_random", "test_family_prior", "test_by_family",
            "leave_family_out", "recorded_price_note", "families"]
    (OUT / "offline.json").write_text(json.dumps({k: off[k] for k in keep if k in off}, separators=(",", ":")))

    pred = pd.read_parquet(ROOT / "results" / f"test_predictions_{PAIR}.parquet")
    slim = D.load_slim().set_index("sample_id")
    fams = sorted(pred.family.unique())
    fam_idx = pred.family.map({f: i for i, f in enumerate(fams)}).to_numpy()
    rng = np.random.default_rng(0)
    text: dict[str, str] = {}
    for f in REDISTRIBUTABLE:
        ids = np.flatnonzero(pred.family.to_numpy() == f)
        for i in rng.permutation(ids)[:16]:
            sid = pred.sample_id.iloc[int(i)]
            text[str(int(i))] = snippet(slim.loc[sid, "prompt"], f)
    r5 = lambda s: [round(float(x), 5) for x in s]  # noqa: E731
    doc = {
        "pair": PAIR, "n": len(pred), "p": r5(pred.p_hard), "sq": r5(pred.s_score), "bq": r5(pred.b_score),
        "sc": [round(float(x), 7) for x in pred.s_cost], "bc": [round(float(x), 7) for x in pred.b_cost],
        "fam": fam_idx.tolist(), "families": fams, "text": text,
        "attribution": {f: REDISTRIBUTABLE[f] for f in REDISTRIBUTABLE},
        "note": "Derived numbers from RouterBench (Hu et al., 2024). Prompt text only for sources with permissive licences.",
    }
    (OUT / "eval.json").write_text(json.dumps(doc, separators=(",", ":")))

    live = ROOT / "results" / "live.json"
    if live.exists():
        (OUT / "live.json").write_text(live.read_text())
    for f in OUT.glob("*.json"):
        print(f"{f.name}: {f.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
