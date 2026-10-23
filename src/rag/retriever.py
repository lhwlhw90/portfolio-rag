"""Hybrid retrieval: dense cosine + BM25, fused with weighted Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .embeddings import Embedder, tokenize
from .store import StoredChunk, VectorStore


@dataclass
class Hit:
    chunk: StoredChunk
    score: float          # fused score
    vector_score: float | None = None
    bm25_score: float | None = None


class HybridRetriever:
    def __init__(self, store: VectorStore, embedder: Embedder, alpha: float = 0.5, rrf_k: int = 60):
        self.store, self.embedder, self.alpha, self.rrf_k = store, embedder, alpha, rrf_k
        self._bm25: BM25Okapi | None = None
        self._bm25_n = -1

    def _bm25_index(self) -> BM25Okapi | None:
        if self._bm25_n != len(self.store.chunks):  # rebuild lazily when the corpus changes
            corpus = [tokenize(c.text) or ["<empty>"] for c in self.store.chunks]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_n = len(self.store.chunks)
        return self._bm25

    def retrieve(self, query: str, k: int = 4) -> list[Hit]:
        if not self.store.chunks or not query.strip():
            return []
        pool = max(k * 4, 20)
        dense = self.store.vector_search(self.embedder.embed_query(query), pool)
        fused: dict[int, float] = {}
        vs: dict[int, float] = {}
        bs: dict[int, float] = {}
        for rank, (i, s) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + self.alpha / (self.rrf_k + rank + 1)
            vs[i] = s
        bm25 = self._bm25_index()
        q_tokens = tokenize(query)
        if bm25 is not None and q_tokens and self.alpha < 1.0:
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
            for rank, i in enumerate(ranked):
                if scores[i] <= 0:
                    break
                fused[i] = fused.get(i, 0.0) + (1 - self.alpha) / (self.rrf_k + rank + 1)
                bs[i] = float(scores[i])
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.store.chunks[i], fused[i], vs.get(i), bs.get(i)) for i in best]
