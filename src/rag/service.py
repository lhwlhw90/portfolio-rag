"""RAGService wires ingest -> embed -> store -> retrieve -> answer."""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Iterator

import httpx

from .answerer import Answer, Answerer, ExtractiveAnswerer, OpenAICompatAnswerer, stream_answer
from .config import Settings
from .embeddings import Embedder, build_embedder
from .ingest import SUPPORTED_EXTS, chunk_pages, parse_document
from .retriever import Hit, HybridRetriever
from .store import VectorStore


class EmptyDocument(ValueError):
    pass


class UnknownAnswerer(ValueError):
    pass


class RAGService:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, answerer: Answerer | None = None):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings.embedding_backend, settings.embedding_model,
                                                   settings.model_dir)
        self.store = VectorStore(settings.db_path, self.embedder.name, self.embedder.dim)
        self.retriever = HybridRetriever(self.store, self.embedder, alpha=settings.hybrid_alpha)
        self.extractive = ExtractiveAnswerer(embedder=self.embedder, min_score=settings.resolved_min_score)
        self._llm: OpenAICompatAnswerer | None = None
        if answerer is not None:
            self.answerer = answerer
        elif settings.llm_backend == "extractive":
            self.answerer = self.extractive
        elif settings.llm_backend == "openai":
            self.answerer = self.llm
        else:
            raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}' (use 'extractive' or 'openai')")
        self.files_dir.mkdir(parents=True, exist_ok=True)

    # ---- answerers ----
    @property
    def llm(self) -> OpenAICompatAnswerer:
        """The OpenAI-compatible answerer is always available per request, even when the
        server default is extractive; unreachable servers fall back to extractive."""
        if self._llm is None:
            s = self.settings
            self._llm = OpenAICompatAnswerer(s.llm_base_url, s.llm_model, s.llm_api_key, s.llm_timeout,
                                             fallback=self.extractive, gate=self.extractive if s.llm_gate else None)
        return self._llm

    def pick_answerer(self, name: str | None) -> Answerer:
        if name in (None, "", "default"):
            return self.answerer
        if name == "extractive":
            return self.extractive
        if name == "llm":
            return self.llm
        raise UnknownAnswerer(name)

    def llm_available(self, timeout: float = 1.0) -> bool:
        try:
            return httpx.get(f"{self.settings.llm_base_url.rstrip('/')}/models", timeout=timeout).status_code == 200
        except Exception:
            return False

    # ---- documents ----
    @property
    def files_dir(self) -> Path:
        return self.settings.data_dir / "files"

    def file_path(self, doc_id: str) -> Path | None:
        doc = self.store.get_document(doc_id)
        if doc is None:
            return None
        p = self.files_dir / f"{doc_id}{Path(doc['name']).suffix.lower()}"
        return p if p.exists() else None

    def ingest(self, filename: str, data: bytes) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        existing = self.store.find_by_hash(sha)
        if existing:
            return {**existing, "duplicate": True}
        pages = parse_document(filename, data)
        doc_id = uuid.uuid4().hex[:12]
        chunks = chunk_pages(doc_id, filename, pages, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise EmptyDocument(f"No extractable text in '{filename}' (scanned PDF? OCR is not supported yet)")
        emb = self.embedder.embed_documents([c.text for c in chunks])
        suffix = Path(filename).suffix.lower()
        assert suffix in SUPPORTED_EXTS
        (self.files_dir / f"{doc_id}{suffix}").write_bytes(data)  # kept for previews / page rendering
        self.store.add_document(doc_id, filename, sha, len(pages), chunks, emb)
        return {"doc_id": doc_id, "name": filename, "pages": len(pages), "n_chunks": len(chunks), "duplicate": False}

    def delete(self, doc_id: str) -> bool:
        path = self.file_path(doc_id)
        ok = self.store.delete_document(doc_id)
        if ok and path is not None:
            path.unlink(missing_ok=True)
        return ok

    # ---- QA ----
    def _result(self, question: str, mode: str, ans: Answer, t0: float) -> dict:
        return {"question": question, "answer": ans.answer, "backend": ans.backend, "refused": ans.refused,
                "refusal_reason": ans.refusal_reason, "refusal_detail": ans.refusal_detail,
                "confidence": ans.confidence, "min_score": self.extractive.min_score,
                "citation_mode": ans.citation_mode, "mode": mode,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
                "citations": [asdict(c) for c in ans.citations]}

    def retrieve(self, question: str, top_k: int | None, mode: str) -> list[Hit]:
        return self.retriever.retrieve(question, k=top_k or self.settings.top_k, mode=mode)

    def ask(self, question: str, top_k: int | None = None, mode: str = "hybrid", answerer: str | None = None) -> dict:
        t0 = time.perf_counter()
        ans_ = self.pick_answerer(answerer)
        hits = self.retrieve(question, top_k, mode)
        return self._result(question, mode, ans_.answer(question, hits), t0)

    def ask_stream(self, question: str, top_k: int | None = None, mode: str = "hybrid",
                   answerer: str | None = None) -> Iterator[tuple[str, object]]:
        """Events: ('retrieval', [...]), ('token', str)*, optional ('reset', None), ('done', result dict)."""
        t0 = time.perf_counter()
        ans_ = self.pick_answerer(answerer)
        hits = self.retrieve(question, top_k, mode)
        yield "retrieval", [{"doc_name": h.chunk.doc_name, "page": h.chunk.page, "chunk_id": h.chunk.chunk_id,
                             "score": round(h.score, 6)} for h in hits]
        for kind, payload in stream_answer(ans_, question, hits):
            if kind == "answer":
                yield "done", self._result(question, mode, payload, t0)
            else:
                yield kind, payload
