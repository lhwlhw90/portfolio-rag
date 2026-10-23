"""RAGService wires ingest -> embed -> store -> retrieve -> answer."""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import asdict

from .answerer import Answer, Answerer, build_answerer
from .config import Settings
from .embeddings import Embedder, build_embedder
from .ingest import chunk_pages, parse_document
from .retriever import HybridRetriever
from .store import VectorStore


class EmptyDocument(ValueError):
    pass


class RAGService:
    def __init__(self, settings: Settings, embedder: Embedder | None = None, answerer: Answerer | None = None):
        self.settings = settings
        self.embedder = embedder or build_embedder(settings.embedding_backend, settings.embedding_model,
                                                   settings.model_dir)
        self.store = VectorStore(settings.db_path, self.embedder.name, self.embedder.dim)
        self.retriever = HybridRetriever(self.store, self.embedder, alpha=settings.hybrid_alpha)
        self.answerer = answerer or build_answerer(settings, self.embedder)

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
        self.store.add_document(doc_id, filename, sha, len(pages), chunks, emb)
        return {"doc_id": doc_id, "name": filename, "pages": len(pages), "n_chunks": len(chunks), "duplicate": False}

    def ask(self, question: str, top_k: int | None = None) -> dict:
        hits = self.retriever.retrieve(question, k=top_k or self.settings.top_k)
        ans: Answer = self.answerer.answer(question, hits)
        return {"question": question, "answer": ans.answer, "backend": ans.backend,
                "citations": [asdict(c) for c in ans.citations]}
