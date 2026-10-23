"""Optional integration test with the real ONNX embedding model.
Run with: make test-slow   (or pytest -m slow). Requires the model in MODEL_DIR
(make download-model) or network access."""
import os
from pathlib import Path

import pytest

from rag.config import Settings, apply_model_cache_env
from rag.service import RAGService
from tests_helpers import SAMPLES

pytestmark = pytest.mark.slow


def test_real_embedding_end_to_end(tmp_path, pdf_bytes):
    model_dir = Path(os.environ.get("MODEL_DIR", Path(__file__).resolve().parents[1] / "models"))
    apply_model_cache_env(model_dir)
    s = Settings(data_dir=tmp_path, model_dir=model_dir, embedding_backend="fastembed", _env_file=None)
    svc = RAGService(s)
    assert svc.embedder.dim == 384
    for name in ("employee_handbook.md", "product_faq.txt"):
        svc.ingest(name, (SAMPLES / name).read_bytes())
    svc.ingest("security_policy.pdf", pdf_bytes)
    # paraphrased query (little lexical overlap) -> needs semantic embedding
    res = svc.ask("How much weight can the robot arm carry?")
    assert "7.5 kg" in res["answer"] and res["citations"][0]["doc_name"] == "product_faq.txt"
    res = svc.ask("Whom do I notify after a data leak and how quickly?")
    assert any(c["doc_name"] == "security_policy.pdf" and c["page"] == 2 for c in res["citations"])
    res = svc.ask("How many vacation days can I carry over?")
    assert "carried over" in res["answer"]
    res = svc.ask("Who are the founders of the company?")
    assert res["citations"] == []  # unanswerable -> "not found" instead of a hallucinated pick
