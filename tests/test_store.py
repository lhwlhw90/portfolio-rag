import numpy as np
import pytest

from rag.embeddings import HashEmbedder
from rag.ingest import Chunk
from rag.store import EmbeddingMismatch, VectorStore


def _chunks(doc_id, texts):
    return [Chunk(doc_id, f"{doc_id}.txt", i, t, None, 0) for i, t in enumerate(texts)]


def test_add_search_persist_delete(tmp_path):
    emb = HashEmbedder()
    db = tmp_path / "s.db"
    s = VectorStore(db, emb.name, emb.dim)
    texts = ["cats purr softly", "rockets launch into orbit", "bread needs yeast"]
    s.add_document("d1", "d1.txt", "h1", 1, _chunks("d1", texts), emb.embed_documents(texts))
    top = s.vector_search(emb.embed_query("rocket launch orbit"), 2)
    assert s.chunks[top[0][0]].text == "rockets launch into orbit"
    assert top[0][1] >= top[1][1]
    s.close()

    s2 = VectorStore(db, emb.name, emb.dim)  # reopen: data persisted
    assert len(s2.chunks) == 3 and s2.matrix.shape == (3, emb.dim)
    assert s2.find_by_hash("h1")["doc_id"] == "d1"
    assert s2.delete_document("d1") and s2.chunks == [] and not s2.delete_document("d1")


def test_embedder_mismatch_detected(tmp_path):
    VectorStore(tmp_path / "s.db", "hash-256", 256).close()
    with pytest.raises(EmbeddingMismatch):
        VectorStore(tmp_path / "s.db", "bge-small", 384)


def test_empty_search(tmp_path):
    s = VectorStore(tmp_path / "s.db", "x", 4)
    assert s.vector_search(np.ones(4, dtype=np.float32), 3) == []


def test_hash_embedder_deterministic_and_normalised():
    a, b = HashEmbedder(), HashEmbedder()
    v = a.embed_documents(["hello world", "foo"])
    assert np.allclose(v, b.embed_documents(["hello world", "foo"]))
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)


def test_resolved_min_score_per_model():
    from rag.config import DEFAULT_MIN_SCORE, MODEL_MIN_SCORES, Settings
    assert Settings(embedding_backend="hash", _env_file=None).resolved_min_score == 0.0
    s = Settings(embedding_model="BAAI/bge-small-zh-v1.5", _env_file=None)
    assert s.resolved_min_score == MODEL_MIN_SCORES["BAAI/bge-small-zh-v1.5"]
    assert Settings(embedding_model="other/model", _env_file=None).resolved_min_score == DEFAULT_MIN_SCORE
    assert Settings(answer_min_score=0.1, _env_file=None).resolved_min_score == 0.1
