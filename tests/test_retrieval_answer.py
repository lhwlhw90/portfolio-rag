import re

import httpx

from rag.answerer import ExtractiveAnswerer, OpenAICompatAnswerer, format_source


def test_hybrid_retrieval_finds_right_doc(loaded_service):
    hits = loaded_service.retriever.retrieve("maximum payload RoboArm", k=3)
    assert hits[0].chunk.doc_name == "product_faq.txt"
    assert hits == sorted(hits, key=lambda h: -h.score)


def test_retrieval_returns_pdf_page(loaded_service):
    hits = loaded_service.retriever.retrieve("how fast must security incidents be reported", k=3)
    top = hits[0].chunk
    assert top.doc_name == "security_policy.pdf" and top.page == 2


def test_answer_has_valid_citations(loaded_service):
    res = loaded_service.ask("How many paid vacation days do full-time employees get?")
    assert "25 days" in res["answer"]
    markers = {int(m) for m in re.findall(r"\[(\d+)\]", res["answer"])}
    idx = {c["index"] for c in res["citations"]}
    assert markers and markers == idx  # every marker resolves, no unused citations
    c = res["citations"][0]
    assert c["doc_name"] == "employee_handbook.md" and c["snippet"] and ":" in c["chunk_id"]
    assert set(c) == {"index", "doc_id", "doc_name", "chunk_id", "page", "snippet", "score"}


def test_pdf_answer_cites_page(loaded_service):
    res = loaded_service.ask("What is the minimum password length?")
    assert "14 characters" in res["answer"]
    pdf_cits = [c for c in res["citations"] if c["doc_name"] == "security_policy.pdf"]
    assert pdf_cits and pdf_cits[0]["page"] == 1


def test_no_documents_no_answer(service):
    res = service.ask("anything?")
    assert res["citations"] == [] and "couldn't find" in res["answer"]


def test_format_source():
    from rag.answerer import Citation
    assert format_source(Citation(1, "d", "a.pdf", "d:0", 3, "s", 0.1)) == "a.pdf, p. 3"
    assert format_source(Citation(1, "d", "a.md", "d:0", None, "s", 0.1)) == "a.md"


def test_duplicate_upload_is_idempotent(loaded_service):
    from tests_helpers import SAMPLES
    r = loaded_service.ingest("again.txt", (SAMPLES / "product_faq.txt").read_bytes())
    assert r["duplicate"] is True and len(loaded_service.store.list_documents()) == 3


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_openai_backend_parses_citations(loaded_service):
    seen = {}

    def handler(req):
        seen["body"] = req.read().decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "Payload is 7.5 kg [1]."}}]})

    ans = OpenAICompatAnswerer("http://llm/v1", "m", client=_mock_client(handler))
    hits = loaded_service.retriever.retrieve("payload", k=3)
    out = ans.answer("payload?", hits)
    assert out.backend == "openai" and [c.index for c in out.citations] == [1]
    assert '<source id=\\"1\\" file=' in seen["body"] and "Question: payload?" in seen["body"]


def test_openai_backend_falls_back_on_error(loaded_service):
    ans = OpenAICompatAnswerer("http://llm/v1", "m", client=_mock_client(lambda r: httpx.Response(500)),
                               fallback=ExtractiveAnswerer())
    out = ans.answer("payload of the RoboArm?", loaded_service.retriever.retrieve("payload RoboArm", k=3))
    assert out.backend.startswith("openai->fallback") and out.citations


def test_extractive_skips_questions_and_orders_by_reading():
    from rag.retriever import Hit
    from rag.store import StoredChunk
    ch = StoredChunk("d:0", "d", "faq.txt", 0, None,
                     "How long is the warranty? The warranty lasts three years for parts. Batteries are covered for one year.")
    out = ExtractiveAnswerer().answer("warranty length years", [Hit(ch, 1.0)])
    assert not out.answer.startswith("How long") and "three years" in out.answer
    assert out.answer.endswith("[1]") and out.citations[0].doc_name == "faq.txt"
    assert out.citations[0].snippet.startswith("The warranty lasts three years")  # snippet = supporting sentence


def test_split_sentences_joins_hard_wraps():
    from rag.answerer import split_sentences
    text = ("# Leave\nUnused vacation days of full-time staff can be carried\nover to next year. Max 5 days.\n\n"
            "Section 2\nMore text here.")
    assert split_sentences(text) == ["Leave", "Unused vacation days of full-time staff can be carried over to next year.",
                                     "Max 5 days.",
                                     "Section 2", "More text here."]


def test_min_score_returns_not_found():
    from rag.answerer import NO_ANSWER
    from rag.retriever import Hit
    from rag.store import StoredChunk
    ch = StoredChunk("d:0", "d", "faq.txt", 0, None, "The warranty lasts three years for parts and labour.")
    out = ExtractiveAnswerer(min_score=0.9).answer("warranty years", [Hit(ch, 1.0)])
    assert out.answer == NO_ANSWER and out.citations == []


def test_retrieval_modes(loaded_service):
    import pytest
    for mode in ("vector", "bm25", "hybrid"):
        hits = loaded_service.retriever.retrieve("RoboArm payload", k=2, mode=mode)
        assert 1 <= len(hits) <= 2
    with pytest.raises(ValueError):
        loaded_service.retriever.retrieve("x", mode="nope")


def test_parse_markers_variants():
    from rag.answerer import parse_markers
    assert parse_markers("a [1] b [2][3] c [1, 4] d 【5】") == [1, 2, 3, 4, 5]


def _llm(handler, **kw):
    return OpenAICompatAnswerer("http://llm/v1", "m", client=_mock_client(handler), **kw)


def _reply(text):
    return lambda req: httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


def test_llm_attributes_citations_when_model_omits_markers(loaded_service):
    hits = loaded_service.retriever.retrieve("maximum payload RoboArm", k=3)
    out = _llm(_reply("The RoboArm X2 can lift a maximum payload of 7.5 kg."), fallback=ExtractiveAnswerer()
               ).answer("payload?", hits)
    assert out.citation_mode == "attributed" and out.answer.endswith("]")
    assert out.citations and out.citations[0].doc_name == "product_faq.txt"


def test_llm_refusal_detected(loaded_service):
    hits = loaded_service.retriever.retrieve("CEO", k=3)
    out = _llm(_reply("I don't know based on the documents.")).answer("Who is the CEO?", hits)
    assert out.refused and out.citations == []


def test_llm_invalid_markers_recorded(loaded_service):
    hits = loaded_service.retriever.retrieve("payload", k=2)
    out = _llm(_reply("7.5 kg [1] [9]")).answer("payload?", hits)
    assert out.invalid_markers == [9] and [c.index for c in out.citations] == [1]


def test_llm_gate_refuses_without_calling_model(loaded_service):
    calls = []

    def handler(req):
        calls.append(1)
        return httpx.Response(200, json={"choices": [{"message": {"content": "x [1]"}}]})

    gate = ExtractiveAnswerer(min_score=99.0)  # impossible threshold -> always refuse
    out = _llm(handler, gate=gate).answer("payload?", loaded_service.retriever.retrieve("payload", k=2))
    assert out.refused and out.backend == "openai(gated)" and calls == []


def test_api_exposes_refusal_and_confidence(loaded_service):
    res = loaded_service.ask("How long is the warranty?")
    assert res["refused"] is False and res["confidence"] is not None and res["latency_ms"] >= 0
