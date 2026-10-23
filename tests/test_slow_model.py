"""Optional integration tests with real ONNX embedding models (and a live local LLM).
Run with: make test-slow   (or pytest -m slow). Requires models in MODEL_DIR
(make download-model) or network access. The LLM test is skipped unless an
OpenAI-compatible server (e.g. `make ollama-up`) answers at LLM_BASE_URL."""
import os
from pathlib import Path

import httpx
import pytest

from rag.config import Settings, apply_model_cache_env
from rag.service import RAGService

pytestmark = pytest.mark.slow
ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = Path(os.environ.get("MODEL_DIR", ROOT / "models"))


def _service(tmp_path, lang, model, **kw):
    apply_model_cache_env(MODEL_DIR)
    s = Settings(data_dir=tmp_path, model_dir=MODEL_DIR, embedding_backend="fastembed", embedding_model=model,
                 _env_file=None, **kw)
    svc = RAGService(s)
    for p in sorted((ROOT / "sample_docs" / lang).iterdir()):
        svc.ingest(p.name, p.read_bytes())
    return svc


def test_real_embedding_english(tmp_path):
    svc = _service(tmp_path, "en", "BAAI/bge-small-en-v1.5")
    assert svc.embedder.dim == 384
    res = svc.ask("How many vacation days can I carry over?")
    assert "carried over" in res["answer"] and res["citations"][0]["doc_name"] == "employee_handbook.md"
    res = svc.ask("Whom do I notify after a security incident and how quickly?")
    assert any(c["doc_name"] == "security_policy.pdf" and c["page"] == 2 for c in res["citations"])
    res = svc.ask("Who are the founders of the company?")
    assert res["refused"] and res["citations"] == []


def test_real_embedding_chinese(tmp_path):
    svc = _service(tmp_path, "zh", "BAAI/bge-small-zh-v1.5", chunk_size=200, chunk_overlap=30)
    res = svc.ask("报销的钱多久能到账？")
    assert "7个工作日" in res["answer"] and res["citations"][0]["doc_name"] == "报销制度.txt"
    res = svc.ask("公司的CEO是谁？")
    assert res["refused"]
    # Chinese PDF (Noto CJK via Chrome print-to-PDF): answer cites the right page
    res = svc.ask("S7安静模式下的噪音是多少分贝？")
    assert "55分贝" in res["answer"]
    assert any(c["doc_name"] == "S7快速入门指南.pdf" and c["page"] == 3 for c in res["citations"])
    res = svc.ask("故障代码E7表示什么？")
    assert "尘盒" in res["answer"] and res["citations"][0]["page"] == 4
    res = svc.ask("故障代码E5表示什么？")
    assert res["refused"] and res["refusal_reason"] == "unknown_identifier"


def test_live_llm_streaming(tmp_path):
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    try:
        httpx.get(base.rsplit("/v1", 1)[0] + "/api/version", timeout=2).raise_for_status()
    except Exception:
        pytest.skip(f"no local LLM server at {base}")
    svc = _service(tmp_path, "en", "BAAI/bge-small-en-v1.5", llm_base_url=base,
                   llm_model=os.environ.get("LLM_MODEL", "qwen2.5:0.5b"))
    events = list(svc.ask_stream("What is the maximum payload of the RoboArm X2?", answerer="llm"))
    kinds = [k for k, _ in events]
    assert kinds[0] == "retrieval" and kinds.count("token") >= 2 and kinds[-1] == "done"
    assert events[-1][1]["backend"] == "openai" and "7.5" in events[-1][1]["answer"]


def test_live_llm_backend(tmp_path):
    base = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    try:
        httpx.get(base.rsplit("/v1", 1)[0] + "/api/version", timeout=2).raise_for_status()
    except Exception:
        pytest.skip(f"no local LLM server at {base}")
    svc = _service(tmp_path, "en", "BAAI/bge-small-en-v1.5", llm_backend="openai", llm_base_url=base,
                   llm_model=os.environ.get("LLM_MODEL", "qwen2.5:0.5b"))
    res = svc.ask("What is the maximum payload of the RoboArm X2?")
    assert res["backend"] == "openai" and "7.5" in res["answer"]
    assert res["citations"] and res["citations"][0]["doc_name"] == "product_faq.txt"
    # broken endpoint -> graceful fallback to extractive
    svc.answerer.base_url = "http://127.0.0.1:9/v1"
    res = svc.ask("What is the maximum payload of the RoboArm X2?")
    assert res["backend"].startswith("openai->fallback") and "7.5 kg" in res["answer"]
