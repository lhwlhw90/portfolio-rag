"""Persistent vector store: SQLite (metadata + float32 blobs) with an
in-memory numpy matrix for brute-force cosine search. Fine up to ~100k chunks
on CPU; swap for FAISS/Chroma behind the same interface if needed."""
from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .ingest import Chunk

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
  doc_id TEXT PRIMARY KEY, name TEXT NOT NULL, sha256 TEXT NOT NULL UNIQUE,
  pages INTEGER, n_chunks INTEGER, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS chunks (
  chunk_id TEXT PRIMARY KEY, doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
  chunk_index INTEGER, page INTEGER, start_char INTEGER, text TEXT NOT NULL,
  meta TEXT, embedding BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT);
"""


@dataclass
class StoredChunk:
    chunk_id: str
    doc_id: str
    doc_name: str
    chunk_index: int
    page: int | None
    text: str
    heading: str | None = None  # section heading in force at the chunk start (meta["heading"])


class EmbeddingMismatch(RuntimeError):
    pass


class VectorStore:
    def __init__(self, db_path: Path | str, embedder_name: str, dim: int):
        self.db_path = str(db_path)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.executescript(SCHEMA)
        self._check_embedder(embedder_name, dim)
        self._reload()

    def _check_embedder(self, name: str, dim: int) -> None:
        row = self._conn.execute("SELECT v FROM kv WHERE k='embedder'").fetchone()
        sig = json.dumps({"name": name, "dim": dim})
        if row is None:
            self._conn.execute("INSERT INTO kv VALUES ('embedder', ?)", (sig,))
            self._conn.commit()
        elif row[0] != sig:
            raise EmbeddingMismatch(
                f"Store was built with {row[0]} but current embedder is {sig}. "
                "Delete the data dir or re-ingest.")

    def _reload(self) -> None:
        rows = self._conn.execute(
            "SELECT c.chunk_id, c.doc_id, d.name, c.chunk_index, c.page, c.text, c.embedding, c.meta "
            "FROM chunks c JOIN documents d USING(doc_id) ORDER BY d.created_at, c.doc_id, c.chunk_index"
        ).fetchall()
        self.chunks = [StoredChunk(r[0], r[1], r[2], r[3], r[4], r[5], json.loads(r[7] or "{}").get("heading"))
                       for r in rows]
        self.matrix = (np.stack([np.frombuffer(r[6], dtype=np.float32) for r in rows])
                       if rows else np.zeros((0, 0), dtype=np.float32))

    # ---- documents ----
    def find_by_hash(self, sha256: str) -> dict | None:
        r = self._conn.execute("SELECT doc_id, name, pages, n_chunks, created_at FROM documents WHERE sha256=?",
                               (sha256,)).fetchone()
        return None if r is None else dict(zip(("doc_id", "name", "pages", "n_chunks", "created_at"), r))

    def add_document(self, doc_id: str, name: str, sha256: str, pages: int,
                     chunks: list[Chunk], embeddings: np.ndarray) -> None:
        assert len(chunks) == len(embeddings)
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("INSERT INTO documents (doc_id, name, sha256, pages, n_chunks) VALUES (?,?,?,?,?)",
                        (doc_id, name, sha256, pages, len(chunks)))
            cur.executemany(
                "INSERT INTO chunks VALUES (?,?,?,?,?,?,?,?)",
                [(c.chunk_id, doc_id, c.chunk_index, c.page, c.start_char, c.text, json.dumps(c.meta),
                  np.asarray(e, dtype=np.float32).tobytes()) for c, e in zip(chunks, embeddings)])
            self._conn.commit()
            self._reload()

    def list_documents(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT doc_id, name, pages, n_chunks, created_at FROM documents ORDER BY created_at").fetchall()
        return [dict(zip(("doc_id", "name", "pages", "n_chunks", "created_at"), r)) for r in rows]

    def get_document(self, doc_id: str) -> dict | None:
        r = self._conn.execute("SELECT doc_id, name, pages, n_chunks, created_at FROM documents WHERE doc_id=?",
                               (doc_id,)).fetchone()
        return None if r is None else dict(zip(("doc_id", "name", "pages", "n_chunks", "created_at"), r))

    def get_chunk(self, chunk_id: str) -> StoredChunk | None:
        return next((c for c in self.chunks if c.chunk_id == chunk_id), None)

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            n = self._conn.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,)).rowcount
            self._conn.commit()
            self._reload()
            return n > 0

    # ---- search ----
    def vector_search(self, query_vec: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Returns (row_index, cosine_score) best-first."""
        if not self.chunks:
            return []
        scores = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        k = min(k, len(scores))
        idx = np.argpartition(-scores, k - 1)[:k]
        idx = idx[np.argsort(-scores[idx])]
        return [(int(i), float(scores[i])) for i in idx]

    def close(self) -> None:
        self._conn.close()
