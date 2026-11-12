"""Embed every RouterBench prompt (MiniLM, CPU) into data/emb_routerbench.npy; resumable in chunks.

    python scripts/embed_routerbench.py
"""

from __future__ import annotations

from pathlib import Path

import torch

from modelrouter import data as D
from modelrouter import features as F

if __name__ == "__main__":
    torch.set_num_threads(3)
    df = D.load_slim()
    F.featurize(df.prompt.tolist(), cache=Path("data/emb_routerbench.npy"), progress=True)
    print("done", flush=True)
