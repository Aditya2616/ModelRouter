"""RouterBench download, integrity check, safe load and conversion to a slim Parquet file.

RouterBench (Hu et al., 2024) records, for every prompt, the score and cost of 11 LLMs. The
dataset card carries no licence (the accompanying code is MIT) so the data is downloaded at
runtime and never redistributed from this repository.
"""

from __future__ import annotations

import ast
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .safepickle import safe_load

URL = "https://huggingface.co/datasets/withmartian/routerbench/resolve/main/routerbench_0shot.pkl"
SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
DATA_DIR = Path("data")
RAW = DATA_DIR / "routerbench_0shot.pkl"
SLIM = DATA_DIR / "routerbench_slim.parquet"

PAIRS = {
    "gpt35_vs_gpt4": ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"),
    "mixtral_vs_gpt4": ("mistralai/mixtral-8x7b-chat", "gpt-4-1106-preview"),
}


def model_columns(df: pd.DataFrame) -> list[str]:
    """Models in a RouterBench frame: every column that has a matching `<model>|total_cost` column."""
    return sorted(c for c in df.columns if f"{c}|total_cost" in df.columns)


def short(model: str) -> str:
    return model.split("/")[-1].replace("-", "_").replace(".", "_").lower()


def family(eval_name: str) -> str:
    """Collapse the 86 sub-benchmark names into a handful of source families."""
    n = eval_name.lower()
    if n.startswith("mmlu"):
        return "mmlu"
    if n in {"hellaswag", "arc-challenge", "winogrande", "mbpp"}:
        return n
    if n == "grade-school-math":
        return "gsm8k"
    if n.startswith("mtbench") or n.startswith("mt-bench"):
        return "mt-bench"
    return "other"


def clean_prompt(p) -> str:
    """RouterBench stores each prompt as the string form of a one-element list; return the real text."""
    if isinstance(p, (list, tuple)):
        return "\n".join(str(x) for x in p)
    s = str(p)
    if s[:2] in ("['", '["') and s[-2:] in ("']", '"]'):
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return s
        if isinstance(v, (list, tuple)):
            return "\n".join(str(x) for x in v)
    return s


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(force: bool = False) -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    if RAW.exists() and not force and sha256_of(RAW) == SHA256:
        return RAW
    print(f"downloading {URL} (about 100 MB)")
    urllib.request.urlretrieve(URL, RAW)
    got = sha256_of(RAW)
    if got != SHA256:
        RAW.unlink()
        raise RuntimeError(f"checksum mismatch for RouterBench file: {got} != {SHA256}")
    return RAW


def build_slim(raw: Path = RAW, out: Path = SLIM) -> pd.DataFrame:
    """Load through the allow-list unpickler and keep only what routing needs (no responses)."""
    with open(raw, "rb") as f:
        df = safe_load(f)
    keep = pd.DataFrame({"sample_id": df["sample_id"].astype(str), "prompt": df["prompt"].map(clean_prompt),
                         "task": df["eval_name"]})
    keep["family"] = keep["task"].map(family)
    for m in model_columns(df):
        keep[f"score__{short(m)}"] = df[m].astype("float32")
        keep[f"cost__{short(m)}"] = df[f"{m}|total_cost"].astype("float64")
    keep = keep.drop_duplicates("prompt").reset_index(drop=True)
    out.parent.mkdir(exist_ok=True)
    keep.to_parquet(out, index=False)
    return keep


def load_slim() -> pd.DataFrame:
    if not SLIM.exists():
        build_slim(fetch())
    return pd.read_parquet(SLIM)


def pair_arrays(df: pd.DataFrame, pair: str = "gpt35_vs_gpt4") -> dict[str, np.ndarray]:
    s, b = PAIRS[pair]
    return {
        "s_score": df[f"score__{short(s)}"].to_numpy(dtype=float),
        "b_score": df[f"score__{short(b)}"].to_numpy(dtype=float),
        "s_cost": df[f"cost__{short(s)}"].to_numpy(dtype=float),
        "b_cost": df[f"cost__{short(b)}"].to_numpy(dtype=float),
    }
