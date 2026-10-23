"""Hybrid retrieval: dense cosine + BM25, fused with weighted Reciprocal Rank Fusion."""
from __future__ import annotations

from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from .embeddings import Embedder
from .text import content_tokens
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
            corpus = [content_tokens(c.text) or ["<empty>"] for c in self.store.chunks]
            self._bm25 = BM25Okapi(corpus) if corpus else None
            self._bm25_n = len(self.store.chunks)
        return self._bm25

    def retrieve(self, query: str, k: int = 4, mode: str = "hybrid", expansions: list[str] | None = None,
                 expansion_weight: float = 1.0) -> list[Hit]:
        """mode: 'hybrid' (weighted RRF), 'vector' (dense only) or 'bm25' (lexical only).
        expansions: extra queries (rewrites / HyDE passage / PRF terms, see rag.query); their hybrid
        rankings are fused with the original one by RRF (original weight 1, expansions share
        `expansion_weight`). Hybrid mode only."""
        if mode not in ("hybrid", "vector", "bm25"):
            raise ValueError(f"unknown retrieval mode {mode!r}")
        if not self.store.chunks or not query.strip():
            return []
        if mode == "vector":
            return [Hit(self.store.chunks[i], s, vector_score=s)
                    for i, s in self.store.vector_search(self.embedder.embed_query(query), k)]
        if mode == "bm25":
            bm25, q = self._bm25_index(), content_tokens(query)
            if bm25 is None or not q:
                return []
            scores = bm25.get_scores(q)
            ranked = [i for i in sorted(range(len(scores)), key=lambda i: -scores[i])[:k] if scores[i] > 0]
            return [Hit(self.store.chunks[i], float(scores[i]), bm25_score=float(scores[i])) for i in ranked]
        pool = max(k * 4, 20)
        fused, vs, bs = self._hybrid(query, pool)
        extra = [e for e in (expansions or []) if e and e.strip()]
        if extra:
            base = sorted(fused, key=lambda i: -fused[i])
            total = {i: 1.0 / (self.rrf_k + r + 1) for r, i in enumerate(base)}
            w = expansion_weight / len(extra)
            for e in extra:
                f2, v2, b2 = self._hybrid(e, pool)
                for r, i in enumerate(sorted(f2, key=lambda i: -f2[i])):
                    total[i] = total.get(i, 0.0) + w / (self.rrf_k + r + 1)
                for i, s in v2.items():  # keep the original query's component scores when present
                    vs.setdefault(i, s)
                for i, s in b2.items():
                    bs.setdefault(i, s)
            fused = total
        best = sorted(fused, key=lambda i: -fused[i])[:k]
        return [Hit(self.store.chunks[i], fused[i], vs.get(i), bs.get(i)) for i in best]

    def _hybrid(self, query: str, pool: int) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
        dense = self.store.vector_search(self.embedder.embed_query(query), pool)
        fused: dict[int, float] = {}
        vs: dict[int, float] = {}
        bs: dict[int, float] = {}
        for rank, (i, s) in enumerate(dense):
            fused[i] = fused.get(i, 0.0) + self.alpha / (self.rrf_k + rank + 1)
            vs[i] = s
        bm25 = self._bm25_index()
        q_tokens = content_tokens(query)
        if bm25 is not None and q_tokens and self.alpha < 1.0:
            scores = bm25.get_scores(q_tokens)
            ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
            for rank, i in enumerate(ranked):
                if scores[i] <= 0:
                    break
                fused[i] = fused.get(i, 0.0) + (1 - self.alpha) / (self.rrf_k + rank + 1)
                bs[i] = float(scores[i])
        return fused, vs, bs
