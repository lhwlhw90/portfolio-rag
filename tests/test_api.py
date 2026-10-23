from fastapi.testclient import TestClient

from rag.api import create_app
from tests_helpers import SAMPLES


def _client(settings, service):
    return TestClient(create_app(settings, service))


def test_full_api_flow(settings, service, pdf_bytes):
    c = _client(settings, service)
    assert c.get("/health").json()["documents"] == 0
    r = c.post("/documents", files={"file": ("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes(), "text/plain")})
    assert r.status_code == 201 and r.json()["n_chunks"] >= 1
    r = c.post("/documents", files={"file": ("security_policy.pdf", pdf_bytes, "application/pdf")})
    assert r.status_code == 201 and r.json()["pages"] == 2
    docs = c.get("/documents").json()
    assert {d["name"] for d in docs} == {"product_faq.txt", "security_policy.pdf"}

    r = c.post("/ask", json={"question": "How long is the warranty?"})
    body = r.json()
    assert r.status_code == 200 and "3-year" in body["answer"]
    assert body["citations"][0]["doc_name"] == "product_faq.txt" and body["backend"] == "extractive"

    pdf_id = next(d["doc_id"] for d in docs if d["name"] == "security_policy.pdf")
    assert c.delete(f"/documents/{pdf_id}").status_code == 204
    assert c.delete(f"/documents/{pdf_id}").status_code == 404
    assert len(c.get("/documents").json()) == 1


def test_upload_rejects_bad_files(settings, service):
    c = _client(settings, service)
    assert c.post("/documents", files={"file": ("x.docx", b"abc", "application/octet-stream")}).status_code == 415
    assert c.post("/documents", files={"file": ("empty.txt", b"   ", "text/plain")}).status_code == 422


def test_ask_validation(settings, service):
    c = _client(settings, service)
    assert c.post("/ask", json={"question": ""}).status_code == 422
    assert c.post("/ask", json={"question": "hi", "top_k": 99}).status_code == 422


def test_index_page_served(settings, service):
    r = _client(settings, service).get("/")
    assert r.status_code == 200 and "Portfolio RAG" in r.text
