"""Check that no prompt can sit on both sides of a train/test split, and write results/duplicates.json.

Prompts are compared by a normalised key (lower case, single spaces; for GSM8K the real question after the
few-shot block, because every GSM8K prompt starts with the same solved examples). If every key is unique, a
split by row is already a split by prompt.

    python scripts/check_duplicates.py
"""

from __future__ import annotations

import json
from pathlib import Path

from modelrouter import data as D

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    df = D.load_slim()
    keys = [D.prompt_key(p, f) for p, f in zip(df.prompt, df.family, strict=True)]
    dup = df.assign(key=keys).groupby("family").key.agg(lambda s: int(len(s) - s.nunique()))
    out = {"n_prompts": len(df), "n_unique_keys": len(set(keys)), "duplicate_rows": len(keys) - len(set(keys)),
           "duplicate_rows_by_family": dup.to_dict(),
           "note": "key = normalised question text; exact duplicate prompts were already dropped when the data was loaded"}
    (ROOT / "results" / "duplicates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    assert out["duplicate_rows"] == 0, "duplicate prompts could leak across the split: group the split by key"


if __name__ == "__main__":
    main()
