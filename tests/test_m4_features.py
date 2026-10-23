"""M4: heading context, answerability judges (policies, refusal reason), query expansion
(PRF / LLM rewrite / HyDE with graceful fallback, multi-query fusion) and the dev-only
calibration helpers used by the eval."""
import json

import httpx
import pytest
from fastapi.testclient import TestClient

from rag import eval_ext
from rag.answerer import ExtractiveAnswerer, with_heading
from rag.api import create_app
from rag.embeddings import HashEmbedder
from rag.ingest import chunk_pages, parse_document
from rag.query import QueryExpander, prf_terms
from rag.retriever import Hit
from rag.store import StoredChunk
from rag.text import last_heading, sentences_with_headings, split_sentences
from tests_helpers import SAMPLES

RELEASE_NOTES = (SAMPLES.parent / "en" / "release_notes.md")


def _hit(text, heading=None, rank_score=1.0, doc="notes.md", idx=0):
    return Hit(StoredChunk(f"d:{idx}", "d", doc, idx, None, text, heading), rank_score)


# ---------------------------------------------------------------- heading context
def test_sentences_with_headings_matches_split_sentences():
    text = RELEASE_NOTES.read_text()
    pairs = sentences_with_headings(text)
    assert [s for s, _ in pairs] == split_sentences(text)  # same order => (rank, order) indices line up
    d = dict(pairs)
    assert d["First release for the X2 hardware."] == "Version 2.0 (2025-07-01)"
    assert d["Version 2.0 (2025-07-01)"] is None           # a heading has no heading of its own
    assert last_heading("intro line that is long enough to be a sentence here.\n\n## Setup\nbody text.") == "Setup"
    assert last_heading("plain sentence text without headings at all.", inherited="Prev") == "Prev"


def test_with_heading_rules():
    h, s = "Version 2.0 (2025-07-01)", "First release for the X2 hardware."
    assert with_heading(s, h, "When was the first X2 firmware released?") == f"{h}: {s}"
    assert with_heading(s, h, "Which hardware did the first firmware support?") == s     # not temporal, no new terms
    assert with_heading(s, None, "When was it released?") == s
    # numbers that come from the question itself (the 'X2' in a title) don't count as new information
    assert with_heading("It has a warranty.", "RoboArm X2 FAQ", "What warranty does the X2 have?") == "It has a warranty."
    # >= 2 question terms supplied by the heading
    assert with_heading("You receive 16 weeks.", "Parental leave", "How long is parental leave?") == \
        "Parental leave: You receive 16 weeks."
    assert with_heading("升级约需10分钟。", "版本 1.2（2026-01-05）", "版本1.2什么时候发布？") == "版本 1.2（2026-01-05）：升级约需10分钟。"


def test_chunk_meta_heading_is_inherited_and_stored(service):
    text = RELEASE_NOTES.read_text()
    chunks = chunk_pages("d", "release_notes.md", parse_document("release_notes.md", text.encode()), 160, 20)
    later = [c for c in chunks if "First release for the X2" in c.text and not c.text.lstrip().startswith("##")]
    assert later and later[0].meta.get("heading") == "Version 2.0 (2025-07-01)"
    # the 160-char window cuts through '## Version 2.0 ...': the fragment must not replace the full heading
    pairs = dict(sentences_with_headings(later[0].text, later[0].meta["heading"]))
    assert pairs["First release for the X2 hardware."] == "Version 2.0 (2025-07-01)"
    service.ingest("release_notes.md", text.encode())   # store round-trip (meta JSON -> StoredChunk.heading)
    s = service.settings
    expected = [c.meta.get("heading") for c in chunk_pages("x", "r.md", parse_document("r.md", text.encode()),
                                                          s.chunk_size, s.chunk_overlap)]
    assert [c.heading for c in service.store.chunks] == expected and any(expected)


def test_extractive_answer_carries_heading_from_chunk_meta():
    # the chunk starts mid-section: the date only exists in the inherited heading
    hit = _hit("First release for the X2 hardware. Maximum speed raised to 1.2 m/s at the tool flange.",
               heading="Version 2.0 (2025-07-01)")
    q = "When was the first X2 firmware released?"
    on = ExtractiveAnswerer(embedder=HashEmbedder()).answer(q, [hit])
    off = ExtractiveAnswerer(embedder=HashEmbedder(), heading_context=False).answer(q, [hit])
    assert "2025-07-01" in on.answer and "2025-07-01" not in off.answer
    assert on.citations[0].snippet.startswith("First release")   # snippet stays the verbatim sentence


# ---------------------------------------------------------------- answerability judge
class FakeJudge:
    name = "fake"

    def __init__(self, score):
        self.value, self.calls = score, 0

    def score(self, question, sentences):
        self.calls += 1
        return self.value


HIT = _hit("Full-time employees receive 25 days of paid vacation per year.", doc="handbook.md")
Q = "How many vacation days do full-time employees get?"


@pytest.mark.parametrize("policy,min_score,judge_score,refused,reason", [
    ("and", 0.0, 0.9, False, None),
    ("and", 0.0, 0.1, True, "not_supported"),
    ("and", 9.9, 0.9, True, "low_confidence"),     # cosine fails first; judge not consulted
    ("replace", 9.9, 0.9, False, None),            # judge replaces the cosine threshold
    ("replace", 0.0, 0.1, True, "not_supported"),
    ("or", 9.9, 0.9, False, None),
    ("or", 0.0, 0.1, False, None),
    ("or", 9.9, 0.1, True, "low_confidence"),
])
def test_judge_policies(policy, min_score, judge_score, refused, reason):
    j = FakeJudge(judge_score)
    a = ExtractiveAnswerer(embedder=HashEmbedder(), min_score=min_score, judge=j, judge_min=0.5, judge_policy=policy)
    ans = a.answer(Q, [HIT])
    assert ans.refused is refused and ans.refusal_reason == reason
    if policy == "and" and min_score > 1:
        assert j.calls == 0 and ans.support_score is None
    else:
        assert j.calls == 1 and ans.support_score == judge_score
    if reason == "not_supported":
        assert "fake" in ans.refusal_detail


def test_judge_policy_validation_and_zh_detail():
    with pytest.raises(ValueError):
        ExtractiveAnswerer(judge_policy="maybe")
    a = ExtractiveAnswerer(embedder=HashEmbedder(), judge=FakeJudge(0.1), judge_min=0.5)
    ans = a.answer("全职员工有多少天带薪年假？", [_hit("全职员工每年享有25天带薪年假，可以结转到下一年。")])
    assert ans.refusal_reason == "not_supported" and "可答性模型" in ans.refusal_detail


def test_service_and_api_expose_judge_and_expansion(settings, service):
    service.extractive.judge, service.extractive.judge_min = FakeJudge(0.2), 0.5
    service.ingest("product_faq.txt", (SAMPLES / "product_faq.txt").read_bytes())
    c = TestClient(create_app(settings, service))
    r = c.post("/ask", json={"question": "How long is the X2 warranty?"}).json()
    assert r["refused"] and r["refusal_reason"] == "not_supported" and r["support_score"] == 0.2
    assert r["query_expansion"] == "off"
    h = c.get("/health").json()
    assert h["answerability_judge"] == "none" and h["heading_context"] is True


def test_build_judge_unknown():
    from rag.answerability import build_judge
    assert build_judge("none", None) is None
    with pytest.raises(ValueError):
        build_judge("bogus", None)


# ---------------------------------------------------------------- query expansion
def _llm_client(reply=None, fail=False, seen=None):
    def handler(req):
        if fail:
            return httpx.Response(500)
        body = json.loads(req.content)
        if seen is not None:
            seen.append(body)
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_rewrite_parses_lines_and_caches():
    seen = []
    qe = QueryExpander("rewrite", client=_llm_client("1. lost or stolen laptop\n- report stolen device\n\nfoo bar baz", seen=seen))
    out = qe.expand("Somebody stole my notebook")
    assert out == ["lost or stolen laptop", "report stolen device", "foo bar baz"]
    assert qe.expand("Somebody stole my notebook") == out and len(seen) == 1   # cached
    assert "search queries" in seen[0]["messages"][0]["content"]
    zh = QueryExpander("rewrite", client=_llm_client("笔记本电脑丢失", seen=seen)).expand("电脑被偷了怎么办")
    assert zh == ["笔记本电脑丢失"] and "检索查询" in seen[-1]["messages"][0]["content"]


def test_hyde_and_failures_fall_back():
    assert QueryExpander("hyde", client=_llm_client("Report the theft to IT. A loaner is issued.")).expand("q?") == \
        ["Report the theft to IT. A loaner is issued."]
    assert QueryExpander("rewrite", client=_llm_client(fail=True)).expand("q?") == []
    assert QueryExpander("off").expand("anything") == []
    with pytest.raises(ValueError):
        QueryExpander("magic")


def test_prf_terms_and_mode(loaded_service):
    terms = prf_terms("vacation", ["Unused vacation days can be carried over to the next year."] * 2,
                      ["Unused vacation days can be carried over to the next year.", "Passwords must be long."])
    assert "vacation" not in terms and "carried" in terms
    qe = QueryExpander("prf")
    out = qe.expand("How many vacation days?", loaded_service.retriever)
    assert len(out) == 1 and out[0] and qe.last_ms >= 0


def test_retriever_fuses_expansions(loaded_service):
    r = loaded_service.retriever
    q = "zebra quokka"  # matches nothing lexically
    rank = lambda hits: next(i for i, h in enumerate(hits) if "3-year limited warranty" in h.chunk.text)
    base = r.retrieve(q, k=3)
    full = r.retrieve(q, k=50)
    exp = r.retrieve(q, k=50, expansions=["three year limited warranty for the X2"])
    assert rank(exp) < rank(full)
    heavier = r.retrieve(q, k=50, expansions=["three year limited warranty for the X2"], expansion_weight=3.0)
    assert rank(heavier) <= rank(exp)
    assert [h.chunk.chunk_id for h in r.retrieve(q, k=3, expansions=[])] == [h.chunk.chunk_id for h in base]


def test_service_uses_expander(settings, service):
    calls = []

    class Spy(QueryExpander):
        def expand(self, question, retriever=None):
            calls.append(question)
            return ["password characters"]
    service.expander = Spy("rewrite")
    service.ingest("employee_handbook.md", (SAMPLES / "employee_handbook.md").read_bytes())
    service.ask("zebra quokka?")
    service.ask("zebra quokka?", mode="bm25")      # expansion is hybrid-only
    assert calls == ["zebra quokka?"]


# ---------------------------------------------------------------- eval helpers
def _rows():
    rows = []
    for i, (cos, j, typ) in enumerate([(0.9, 0.9, "factual"), (0.8, 0.2, "factual"), (0.7, 0.8, "paraphrase"),
                                       (0.75, 0.1, "unanswerable"), (0.6, 0.05, "unanswerable")]):
        rows.append({"id": str(i), "type": typ, "cos": cos, "fake": j, "fake_ms": 10.0, "base_ms": 5.0,
                     "rule": None, "correct": typ != "unanswerable"})
    return rows


def test_variant_calibration_on_dev_only():
    dev = _rows()
    m3 = eval_ext.calibrate_variant(dev, {"name": "m3", "policy": "cosine", "rule": True})
    assert eval_ext.variant_metrics(dev, m3)["balanced_accuracy"] < 100      # cos alone can't separate (0.75 una)
    both = eval_ext.calibrate_variant(dev, {"name": "or", "policy": "or", "judge": "fake", "rule": True})
    m = eval_ext.variant_metrics(dev, both)
    assert m["balanced_accuracy"] == 100.0 and m["answer_accuracy"] == 100.0 and m["judge_ms_p50"] == 10.0
    res = eval_ext.eval_answerability({"dev": dev, "test": dev}, ["fake"])
    assert res["chosen"].startswith("cosine OR fake") or res["chosen"].startswith("fake only")
    assert res["auc"]["dev"]["fake"] == 1.0
    ruled = [{**r, "rule": "unknown_identifier"} if r["id"] == "3" else r for r in dev]
    assert eval_ext._decide(ruled[3], {"policy": "cosine", "rule": True, "t_cos": 0.0})


def test_expansion_cache_persists_with_cost(tmp_path):
    path = tmp_path / "qe.json"
    seen = []
    qe = QueryExpander("hyde", model="m", client=_llm_client("passage", seen=seen), cache_path=path)
    assert qe.expand("q1") == ["passage"] and len(seen) == 1
    qe.save_cache()
    rows = json.loads(path.read_text())
    assert rows[0]["question"] == "q1" and rows[0]["ms"] >= 0
    qe2 = QueryExpander("hyde", model="m", client=_llm_client(fail=True), cache_path=path)
    assert qe2.expand("q1") == ["passage"] and qe2.last_cost_ms == rows[0]["ms"]   # no new LLM call


def test_duplicate_sentences_from_overlapping_chunks_are_shown_once():
    s = "Version 2.1 added Modbus TCP support for PLC integration."
    ans = ExtractiveAnswerer(embedder=HashEmbedder()).answer(
        "Which version added Modbus TCP support?", [_hit(s, idx=0), _hit(s + " Other text here today.", idx=1)])
    assert ans.answer.count("Modbus") == 1


def test_auto_defaults_resolve_per_embedding_model(monkeypatch, tmp_path):
    from rag import config
    from rag.config import Settings
    monkeypatch.setitem(config.ANSWERABILITY_DEFAULTS, "m-en", ("nli", "and", 0.61, 0.2))
    monkeypatch.setitem(config.QUERY_EXPANSION_DEFAULTS, "m-en", ("rewrite", "qwen2.5:1.5b", 0.5))
    s = Settings(embedding_model="m-en", data_dir=tmp_path, _env_file=None)
    assert (s.resolved_judge, s.resolved_policy, s.resolved_min_score, s.resolved_judge_min) == ("nli", "and", 0.61, 0.2)
    assert s.resolved_expansion == ("rewrite", "qwen2.5:1.5b", 0.5)
    off = Settings(embedding_model="m-en", data_dir=tmp_path, answerability_judge="none", query_expansion="off",
                   _env_file=None)
    assert off.resolved_judge == "none" and off.resolved_expansion[0] == "off"
    assert off.resolved_min_score == config.DEFAULT_MIN_SCORE           # M3 per-model cosine threshold
    other = Settings(embedding_model="unknown/model", data_dir=tmp_path, _env_file=None)
    assert other.resolved_judge == "none" and other.resolved_expansion[0] == "off"
    explicit = Settings(embedding_model="m-en", data_dir=tmp_path, query_expansion="hyde",
                        query_expansion_model="qwen2.5:0.5b", _env_file=None)
    assert explicit.resolved_expansion == ("hyde", "qwen2.5:0.5b", 1.0)
    hashed = Settings(embedding_backend="hash", embedding_model="m-en", data_dir=tmp_path, _env_file=None)
    assert hashed.resolved_judge == "none" and hashed.resolved_expansion[0] == "off"


def test_expander_circuit_breaker_after_failure():
    seen = []

    def handler(req):
        seen.append(1)
        raise httpx.ConnectError("refused")
    qe = QueryExpander("rewrite", client=httpx.Client(transport=httpx.MockTransport(handler)), cooldown=60)
    assert qe.expand("a?") == [] and qe.expand("b?") == [] and len(seen) == 1   # second call skipped
    qe._down_until = 0
    assert qe.expand("c?") == [] and len(seen) == 2
