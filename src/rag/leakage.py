"""Leakage checks for eval questions (M5): a new dev question must not duplicate or closely
paraphrase any test question.

Two model-free lexical similarities, both language-agnostic:
- char n-gram cosine: TF vectors of character 3-grams of the normalised text (catches
  near-identical wording, e.g. "What does error code E204 mean?" vs "...E205 mean?");
- content-token Jaccard: overlap of non-stop-word tokens (catches re-ordered or re-worded
  questions that ask the same thing with the same key terms).
`scripts/build_dev_extra.py` additionally reports the max embedding cosine (bge-small) as a
semantic check.
"""
from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from pathlib import Path

from .text import content_tokens

CHAR_MAX = 0.6     # reject if char-3-gram cosine >= this
JACCARD_MAX = 0.5  # reject if content-token Jaccard >= this


def _norm(s: str) -> str:
    return re.sub(r"[\s\W_]+", " ", s.lower()).strip()


def char_ngrams(s: str, n: int = 3) -> Counter:
    t = f" {_norm(s)} "
    return Counter(t[i:i + n] for i in range(max(1, len(t) - n + 1)))


def char_cosine(a: str, b: str, n: int = 3) -> float:
    x, y = char_ngrams(a, n), char_ngrams(b, n)
    dot = sum(v * y[k] for k, v in x.items() if k in y)
    nx, ny = math.sqrt(sum(v * v for v in x.values())), math.sqrt(sum(v * v for v in y.values()))
    return dot / (nx * ny) if nx and ny else 0.0


def token_jaccard(a: str, b: str) -> float:
    x, y = set(content_tokens(a)), set(content_tokens(b))
    return len(x & y) / len(x | y) if x | y else 0.0


def nearest(q: str, pool: list[dict]) -> dict:
    """Most similar pool question by each measure: {'char': (score, id, text), 'jaccard': (...)}."""
    best = {"char": (0.0, None, None), "jaccard": (0.0, None, None)}
    for p in pool:
        c, j = char_cosine(q, p["question"]), token_jaccard(q, p["question"])
        if c > best["char"][0]:
            best["char"] = (round(c, 3), p["id"], p["question"])
        if j > best["jaccard"][0]:
            best["jaccard"] = (round(j, 3), p["id"], p["question"])
    return best


def check(new: list[dict], pool: list[dict], char_max: float = CHAR_MAX, jac_max: float = JACCARD_MAX) -> list[str]:
    """Error messages for every new question that is (near-)identical to a pool question,
    plus exact duplicates within `new` itself."""
    errs, seen = [], {}
    for q in new:
        key = _norm(q["question"])
        if key in seen:
            errs.append(f"{q['id']}: duplicate of {seen[key]}")
        seen[key] = q["id"]
        b = nearest(q["question"], pool)
        if b["char"][0] >= char_max:
            errs.append(f"{q['id']}: char-3gram cosine {b['char'][0]} with {b['char'][1]} {b['char'][2]!r}")
        if b["jaccard"][0] >= jac_max:
            errs.append(f"{q['id']}: token Jaccard {b['jaccard'][0]} with {b['jaccard'][1]} {b['jaccard'][2]!r}")
    return errs


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
