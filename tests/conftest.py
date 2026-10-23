import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from make_pdf import SAMPLE_PAGES, make_pdf  # noqa: E402

from rag.config import Settings  # noqa: E402
from rag.embeddings import HashEmbedder  # noqa: E402
from rag.service import RAGService  # noqa: E402

SAMPLES = ROOT / "sample_docs"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models", embedding_backend="hash",
                    chunk_size=300, chunk_overlap=50, top_k=3, answer_min_score=0.0, _env_file=None)


@pytest.fixture
def service(settings):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return RAGService(settings, embedder=HashEmbedder())


@pytest.fixture
def pdf_bytes():
    return make_pdf(SAMPLE_PAGES)


@pytest.fixture
def loaded_service(service, pdf_bytes):
    service.ingest("employee_handbook.md", (SAMPLES / "employee_handbook.md").read_bytes())
    service.ingest("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes())
    service.ingest("security_policy.pdf", pdf_bytes)
    return service
