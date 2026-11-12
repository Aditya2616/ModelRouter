import pickle

import numpy as np
import pandas as pd

from modelrouter import data
from modelrouter.features import LEXICAL_NAMES, lexical


def test_clean_prompt_unwraps_stringified_list():
    raw = r"['Question one\nA) yes\nB) no']"  # backslash-n escapes, exactly as stored in the dataset
    assert data.clean_prompt(raw) == "Question one\nA) yes\nB) no"
    assert data.clean_prompt("plain text") == "plain text"
    assert data.clean_prompt(["a", "b"]) == "a\nb"


def test_family_mapping():
    assert data.family("mmlu-anatomy") == "mmlu"
    assert data.family("grade-school-math") == "gsm8k"
    assert data.family("mtbench-math") == "mt-bench"
    assert data.family("chinese_zodiac") == "other"


def test_lexical_features_shape_and_signal():
    X = lexical(["What is 2 + 2?", "def f(x):\n    return x * 3\n```", "A) one\nB) two\nC) three"])
    assert X.shape == (3, len(LEXICAL_NAMES)) and X.dtype == np.float32
    names = {n: i for i, n in enumerate(LEXICAL_NAMES)}
    assert X[1, names["has_code_fence"]] == 1 and X[0, names["has_code_fence"]] == 0
    assert X[2, names["has_options"]] == 1 and X[0, names["has_math_symbols"]] == 1
    assert X[1, names["n_lines"]] == 2


def test_build_slim_from_synthetic_pickle(tmp_path):
    n = 5
    cols: dict = {"sample_id": [f"s{i}" for i in range(n)], "prompt": [f"['q{i}']" for i in range(n)],
                  "eval_name": ["hellaswag"] * n}
    for m in data.MODELS:
        cols[m] = np.linspace(0, 1, n)
        cols[f"{m}|model_response"] = ["r"] * n
        cols[f"{m}|total_cost"] = np.full(n, 0.001)
    df = pd.DataFrame(cols)
    for c in df.columns:
        if df[c].dtype != np.float64:
            df[c] = df[c].astype(object)
    df.columns = pd.Index(list(df.columns), dtype=object)
    df.index = pd.Index(list(range(n)), dtype=object)
    raw, out = tmp_path / "x.pkl", tmp_path / "slim.parquet"
    raw.write_bytes(pickle.dumps(df))
    slim = data.build_slim(raw, out)
    assert out.exists() and len(slim) == n and slim.prompt.iloc[0] == "q0"
    assert not any("response" in c for c in slim.columns)  # responses are dropped
    assert data.pair_arrays(slim)["b_cost"].shape == (n,)
