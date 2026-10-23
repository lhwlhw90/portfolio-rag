from pathlib import Path

import pytest

from rag.config import Settings
from rag.embeddings import HashEmbedder
from rag.service import RAGService

ZH = Path(__file__).resolve().parents[1] / "sample_docs" / "zh"


@pytest.fixture
def zh_service(tmp_path):
    s = Settings(data_dir=tmp_path, embedding_backend="hash", chunk_size=200, chunk_overlap=30, top_k=3,
                 _env_file=None)
    svc = RAGService(s, embedder=HashEmbedder())
    for p in sorted(ZH.iterdir()):
        svc.ingest(p.name, p.read_bytes())
    return svc


def test_chinese_chunking_respects_size(zh_service):
    assert len(zh_service.store.chunks) > 4
    assert all(len(c.text) <= 200 for c in zh_service.store.chunks)


@pytest.mark.parametrize("mode", ["bm25", "hybrid"])
def test_chinese_bm25_bigram_retrieval(zh_service, mode):
    hits = zh_service.retriever.retrieve("S7的电池容量是多少？", k=3, mode=mode)
    assert hits[0].chunk.doc_name == "扫地机器人S7说明书.txt"


def test_chinese_answer_with_citation(zh_service):
    res = zh_service.ask("一线城市出差住宿标准是多少？")
    assert "600" in res["answer"] and res["citations"][0]["doc_name"] == "报销制度.txt"
    assert res["answer"].rstrip().endswith("]")


def test_chinese_refusal_message(tmp_path):
    s = Settings(data_dir=tmp_path, embedding_backend="hash", _env_file=None)
    res = RAGService(s, embedder=HashEmbedder()).ask("公司的CEO是谁？")
    assert res["refused"] and res["answer"] == "在已上传的文档中没有找到相关信息。"
