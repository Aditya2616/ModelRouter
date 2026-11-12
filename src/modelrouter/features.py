"""Prompt features: sentence embeddings plus cheap lexical statistics."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LEXICAL_NAMES = [
    "log_chars", "log_words", "n_lines", "digit_ratio", "upper_ratio", "punct_ratio",
    "n_questions", "has_code_fence", "has_math_symbols", "has_options", "avg_word_len",
    "max_word_len", "non_ascii_ratio", "n_sentences", "has_number_words", "has_latex",
]
_OPTION_RE = re.compile(r"(^|\n)\s*(\(?[A-Da-d][\).:]|[1-4][\).])\s+\S")
_MATH_RE = re.compile(r"[=+\-*/^<>%]\s*\d|\d\s*[=+\-*/^<>%]")
_NUMWORDS = re.compile(r"\b(twice|half|double|triple|percent|average|total|each|per|ratio|remaining)\b", re.I)


def lexical(prompts: Sequence[str]) -> np.ndarray:
    rows = []
    for p in prompts:
        n = max(len(p), 1)
        words = p.split()
        nw = max(len(words), 1)
        rows.append([
            np.log1p(len(p)), np.log1p(len(words)), p.count("\n"),
            sum(c.isdigit() for c in p) / n, sum(c.isupper() for c in p) / n,
            sum((not c.isalnum()) and (not c.isspace()) for c in p) / n,
            p.count("?"), float("```" in p or "def " in p or "import " in p),
            float(bool(_MATH_RE.search(p))), float(bool(_OPTION_RE.search(p))),
            sum(len(w) for w in words) / nw, max((len(w) for w in words), default=0),
            sum(ord(c) > 127 for c in p) / n, len(re.findall(r"[.!?](\s|$)", p)),
            float(bool(_NUMWORDS.search(p))), float("\(" in p or "\frac" in p or "$" in p),
        ])
    return np.asarray(rows, dtype=np.float32)


_embedder = None


def embedder(device: str = "cpu"):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL, device=device)
        _embedder.max_seq_length = 256
    return _embedder


def shorten(p: str, head: int = 300, tail: int = 600) -> str:
    """The embedder reads about 256 tokens. Long prompts (few-shot templates) carry the actual question at
    the end, so keep the start and the end rather than only the start."""
    return p if len(p) <= head + tail + 20 else p[:head] + " ... " + p[-tail:]


def embed(prompts: Sequence[str], batch_size: int = 64, progress: bool = False) -> np.ndarray:
    vecs = embedder().encode([shorten(p) for p in prompts], batch_size=batch_size, normalize_embeddings=True,
                             show_progress_bar=progress, convert_to_numpy=True)
    return vecs.astype(np.float32)


def featurize(prompts: Sequence[str], cache: Path | None = None, progress: bool = False) -> np.ndarray:
    """[embedding (384) | lexical (16)]. If `cache` exists and matches, reuse the embeddings."""
    if cache is not None and cache.exists():
        e = np.load(cache)
        if len(e) == len(prompts):
            return np.hstack([e, lexical(prompts)])
    e = embed(prompts, progress=progress)
    if cache is not None:
        cache.parent.mkdir(exist_ok=True)
        np.save(cache, e)
    return np.hstack([e, lexical(prompts)])
