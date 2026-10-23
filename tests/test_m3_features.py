"""M3: source preview (files, page images, highlights), refusal reasons, answerer
switch, streaming, Chinese PDF handling, identifier rule."""
import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from rag.answerer import (ExtractiveAnswerer, OpenAICompatAnswerer, best_supporting_sentence, missing_identifiers,
                          stream_answer)
from rag.api import create_app
from rag.ingest import fold_cjk_radicals, parse_document
from rag.text import identifiers
from tests_helpers import SAMPLES

ROOT = Path(__file__).resolve().parents[1]
ZH_PDF = ROOT / "sample_docs" / "zh" / "S7快速入门指南.pdf"


def _client(settings, service):
    return TestClient(create_app(settings, service))


def _sse(text: str) -> list[tuple[str, object]]:
    out = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.splitlines())
        out.append((lines["event"], json.loads(lines["data"])))
    return out


# ---------------------------------------------------------------- text / ingest
def test_identifiers_and_missing_identifiers():
    assert identifiers("max speed of the RoboArm X3? error E402, firmware v2.4.0 or 2.4") == {"x3", "e402", "v2.4.0", "2.4"}
    assert identifiers("after 5 failed attempts") == set()          # plain integers are ignored
    assert identifiers("S9扫地机器人") == {"s9"}                      # CJK neighbours don't block matching
    ctx = "The X2 supports 2.4GHz Wi-Fi. Error code E101 means overcurrent."
    assert missing_identifiers("What does E402 mean on the X2?", ctx) == ["e402"]
    assert missing_identifiers("Does it work with 2.4G networks?", ctx) == []   # prefix match '2.4g'~'2.4ghz'


def test_fold_cjk_radicals():
    # Kangxi radicals (U+2Fxx) and CJK radical supplements (U+2Exx) as emitted by some PDF fonts
    assert fold_cjk_radicals("快速\u2f0a\u2ed4指\u2f70") == "快速入门指示"
    assert fold_cjk_radicals("ASCII，中文。") == "ASCII，中文。"  # full-width punctuation untouched


def test_chinese_pdf_extracts_clean_text_per_page():
    pages = parse_document(ZH_PDF.name, ZH_PDF.read_bytes())
    assert [p.page for p in pages] == [1, 2, 3, 4]
    text = "".join(p.text for p in pages)
    assert "快速入门指南" in pages[0].text and "安静模式下噪音约为55分贝" in pages[2].text
    assert not any(0x2E80 <= ord(c) <= 0x2FDF for c in text)


# ---------------------------------------------------------------- pdf rendering
def test_render_page_highlights_chinese_snippet():
    from rag.pdfview import page_count, render_page
    data = ZH_PDF.read_bytes()
    assert page_count(data) == 4
    png, n = render_page(data, 3, ["安静模式下噪音约为55分贝，适合夜间使用。"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n" and n >= 1
    _, n0 = render_page(data, 3, ["this sentence is not on the page"])
    assert n0 == 0
    with pytest.raises(IndexError):
        render_page(data, 9)


def test_render_page_highlight_spans_wrapped_lines(pdf_bytes):
    from rag.pdfview import render_page
    _, n = render_page(pdf_bytes, 2, ["The on-call security engineer triages incidents … Customer data breaches are escalated"])
    assert n >= 2


# ---------------------------------------------------------------- API
def test_files_chunks_and_page_images(settings, service, pdf_bytes):
    c = _client(settings, service)
    pdf = c.post("/documents", files={"file": ("security_policy.pdf", pdf_bytes, "application/pdf")}).json()
    txt = c.post("/documents", files={"file": ("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes(), "text/plain")}).json()
    r = c.get(f"/documents/{pdf['doc_id']}/file")
    assert r.status_code == 200 and r.headers["content-type"] == "application/pdf" and r.content == pdf_bytes
    r = c.get(f"/documents/{pdf['doc_id']}/pages/2.png", params={"highlight": "within 1 hour of discovery"})
    assert r.status_code == 200 and r.headers["content-type"] == "image/png" and int(r.headers["x-highlights"]) >= 1
    assert c.get(f"/documents/{pdf['doc_id']}/pages/7.png").status_code == 404
    assert c.get(f"/documents/{txt['doc_id']}/pages/1.png").status_code == 400
    assert c.get("/documents/nope/file").status_code == 404

    body = c.post("/ask", json={"question": "How quickly must security incidents be reported?"}).json()
    cit = body["citations"][0]
    ch = c.get(f"/chunks/{cit['chunk_id']}").json()
    assert ch["doc_name"] == cit["doc_name"] and ch["has_file"] is True and cit["snippet"].split(" … ")[0] in ch["text"]
    assert c.get("/chunks/missing:0").status_code == 404

    stored = settings.data_dir / "files" / f"{pdf['doc_id']}.pdf"
    assert stored.exists()
    assert c.delete(f"/documents/{pdf['doc_id']}").status_code == 204 and not stored.exists()


def test_ask_answerer_switch_and_refusal_reason(settings, service):
    c = _client(settings, service)
    c.post("/documents", files={"file": ("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes(), "text/plain")})
    b = c.post("/ask", json={"question": "How long is the warranty?", "answerer": "extractive"}).json()
    assert b["backend"] == "extractive" and b["refused"] is False and b["refusal_reason"] is None and b["mode"] == "hybrid"
    b = c.post("/ask", json={"question": "What is the payload of the RoboArm X3?", "answerer": "extractive"}).json()
    assert b["refused"] and b["refusal_reason"] == "unknown_identifier" and "x3" in b["refusal_detail"]
    assert c.post("/ask", json={"question": "x", "answerer": "gpt5"}).status_code == 422


def test_ask_llm_switch_falls_back_when_server_down(settings, service):
    settings.llm_base_url = "http://127.0.0.1:9/v1"  # nothing listens on the discard port
    c = _client(settings, service)
    c.post("/documents", files={"file": ("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes(), "text/plain")})
    b = c.post("/ask", json={"question": "How long is the warranty?", "answerer": "llm"}).json()
    assert b["backend"].startswith("openai->fallback:extractive") and "3-year" in b["answer"]
    assert c.get("/health").json()["llm_available"] is False


def test_ask_stream_sse(settings, service):
    c = _client(settings, service)
    c.post("/documents", files={"file": ("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes(), "text/plain")})
    r = c.post("/ask/stream", json={"question": "How long is the warranty?", "mode": "bm25"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    ev = _sse(r.text)
    assert ev[0][0] == "retrieval" and ev[0][1][0]["doc_name"] == "product_faq.txt"
    tokens = "".join(p for k, p in ev if k == "token")
    kind, done = ev[-1]
    assert kind == "done" and done["mode"] == "bm25" and tokens == done["answer"] and "3-year" in tokens


# ---------------------------------------------------------------- LLM streaming
def _sse_transport(chunks, fail_after=None):
    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["stream"] is True
        lines = []
        for i, t in enumerate(chunks):
            lines.append("data: " + json.dumps({"choices": [{"delta": {"content": t}}]}))
        lines.append("data: [DONE]")
        return httpx.Response(200, text="\n\n".join(lines) + "\n\n", headers={"content-type": "text/event-stream"})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_llm_stream_tokens_then_postprocessed_answer(loaded_service):
    hits = loaded_service.retriever.retrieve("How long is the warranty?", k=3)
    llm = OpenAICompatAnswerer("http://x/v1", "m", client=_sse_transport(["The warranty ", "is 3 years ", "[1] [1]."]))
    ev = list(llm.stream("How long is the warranty?", hits))
    assert [k for k, _ in ev] == ["token", "token", "token", "answer"]
    ans = ev[-1][1]
    assert ans.answer.count("[1]") == 1 and ans.citations[0].index == 1 and ans.refused is False


def test_llm_stream_falls_back_on_error(loaded_service):
    def boom(request):
        raise httpx.ConnectError("down")
    hits = loaded_service.retriever.retrieve("How long is the warranty?", k=3)
    llm = OpenAICompatAnswerer("http://x/v1", "m", client=httpx.Client(transport=httpx.MockTransport(boom)))
    ev = list(llm.stream("How long is the warranty?", hits))
    assert ev[-1][0] == "answer" and ev[-1][1].backend.startswith("openai->fallback")


def test_stream_answer_for_extractive_reassembles_text(loaded_service):
    hits = loaded_service.retriever.retrieve("How long is the warranty?", k=3)
    ev = list(stream_answer(ExtractiveAnswerer(), "How long is the warranty?", hits))
    assert ev[-1][0] == "answer" and "".join(p for k, p in ev if k == "token") == ev[-1][1].answer


def test_llm_refusal_has_reason_and_snippet_is_sharpened(loaded_service):
    hits = loaded_service.retriever.retrieve("How long is the warranty?", k=3)
    def reply(text):
        return httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json={"choices": [{"message": {"content": text}}]})))
    a = OpenAICompatAnswerer("http://x/v1", "m", client=reply("The documents do not say.")).answer("CEO?", hits)
    assert a.refused and a.refusal_reason == "llm_refused"
    a = OpenAICompatAnswerer("http://x/v1", "m", client=reply("It has a 3-year limited warranty [1].")).answer("warranty?", hits)
    assert "3-year limited warranty" in a.citations[0].snippet and len(a.citations[0].snippet) < 160


def test_best_supporting_sentence():
    chunk = "How long is the warranty?\nThe RoboArm X2 comes with a 3-year limited warranty. Batteries are covered for 12 months."
    assert best_supporting_sentence("Batteries: 12 months", chunk).startswith("Batteries are covered")


def test_refusal_detail_is_localized(loaded_service):
    hits = loaded_service.retriever.retrieve("RoboArm X2", k=3)
    a = ExtractiveAnswerer().answer("RoboArm S9 的 maximum payload 是多少？", hits)
    assert a.refused and a.refusal_reason == "unknown_identifier" and "未找到" in a.refusal_detail
