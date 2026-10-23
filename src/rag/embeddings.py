"""Embedding backends. All return L2-normalised float32 matrices."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .text import content_tokens, tokenize  # noqa: F401  (tokenize re-exported)

def _l2(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (m / norms).astype(np.float32)


class Embedder(Protocol):
    name: str
    dim: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic hashing-trick bag-of-words embedder. No model, no network.
    Used in tests and as an ultra-light fallback."""

    def __init__(self, dim: int = 256):
        self.dim = dim
        self.name = f"hash-{dim}"

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        for tok in content_tokens(text):
            h = int.from_bytes(hashlib.md5(tok.encode()).digest()[:4], "little")
            v[h % self.dim] += 1.0 if (h >> 31) & 1 == 0 else -1.0
        return v

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack([self._vec(t) for t in texts]))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(self._vec(text)[None, :])[0]


class FastEmbedEmbedder:
    """ONNX-runtime embedder via `fastembed` (no torch). Model files are stored
    under MODEL_DIR (explicit cache_dir), never in ~/.cache."""

    def __init__(self, model_name: str, model_dir: Path):
        from fastembed import TextEmbedding

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(model_dir))
        self.dim = int(self.embed_query("dimension probe").shape[0])

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _l2(np.stack(list(self._model.passage_embed(texts))))

    def embed_query(self, text: str) -> np.ndarray:
        return _l2(np.stack(list(self._model.query_embed([text]))))[0]


def build_embedder(backend: str, model_name: str, model_dir: Path) -> Embedder:
    if backend == "hash":
        return HashEmbedder()
    if backend == "fastembed":
        return FastEmbedEmbedder(model_name, model_dir)
    raise ValueError(f"Unknown EMBEDDING_BACKEND '{backend}' (use 'fastembed' or 'hash')")
