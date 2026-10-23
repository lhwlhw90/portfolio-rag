"""M5: judge input with heading context, constrained (false-refusal-capped) selection on dev,
per-language defaults incl. the judge=none fallback, and the dev-extension leakage guards."""
import json
from pathlib import Path

import pytest

from rag import eval_ext
from rag.answerer import ExtractiveAnswerer, judge_text
from rag.config import Settings
from rag.embeddings import HashEmbedder
from rag.leakage import char_cosine, check, sha256, token_jaccard
from rag.retriever import Hit
from rag.store import StoredChunk

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "eval" / "datasets"
H, S = "Version 2.0 (2025-07-01)", "First release for the X2 hardware."
Q = "When was the first X2 firmware released?"


def _rows(path):
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------------------------------------------------------------- judge input
def test_judge_text_modes():
    assert judge_text(S, H, Q, "plain") == S
    assert judge_text(S, H, Q, "heading") == f"{H}: {S}"                      # = what the answer shows
    assert judge_text(S, H, "Which hardware did it support?", "heading") == S  # rule says no heading here
    assert judge_text(S, H, "Which hardware did it support?", "section") == f"{H}: {S}"
    assert judge_text(S, None, Q, "section") == S
    assert judge_text("升级约需10分钟。", "版本 1.2（2026-01-05）", "版本1.2什么时候发布？", "section") == \
        "版本 1.2（2026-01-05）：升级约需10分钟。"
    with pytest.raises(ValueError):
        judge_text(S, H, Q, "bogus")
    with pytest.raises(ValueError):
        ExtractiveAnswerer(judge_input="bogus")


class RecordingJudge:
    name = "rec"

    def __init__(self):
        self.seen = []

    def score(self, question, sentences):
        self.seen.append(list(sentences))
        return 0.9 if any("2025-07-01" in s for s in sentences) else 0.1


@pytest.mark.parametrize("mode,refused", [("plain", True), ("heading", False), ("section", False)])
def test_judge_reads_heading_context(mode, refused):
    """The M4 failure: the date lives only in the heading; a judge reading the bare sentence refuses."""
    hit = Hit(StoredChunk("d:0", "d", "release_notes.md", 0, None,
                          "First release for the X2 hardware. Maximum speed raised to 1.2 m/s at the tool flange.", H), 1.0)
    j = RecordingJudge()
    ans = ExtractiveAnswerer(embedder=HashEmbedder(), judge=j, judge_min=0.5, judge_policy="replace",
                             judge_input=mode).answer(Q, [hit])
    assert ans.refused is refused
    assert any(s.startswith(H) for s in j.seen[0]) is (mode != "plain")


# ---------------------------------------------------------------- per-language defaults
def test_m5_defaults_and_none_fallback(monkeypatch, tmp_path):
    from rag import config
    monkeypatch.setitem(config.ANSWERABILITY_DEFAULTS, "m-en", ("none", None, 0.67, None, None))
    monkeypatch.setitem(config.ANSWERABILITY_DEFAULTS, "m-zh", ("reranker-m3", "and", 0.6, 0.3, "heading"))
    en = Settings(embedding_model="m-en", data_dir=tmp_path, _env_file=None)
    assert (en.resolved_judge, en.resolved_min_score) == ("none", 0.67)
    zh = Settings(embedding_model="m-zh", data_dir=tmp_path, _env_file=None)
    assert (zh.resolved_judge, zh.resolved_policy, zh.resolved_min_score, zh.resolved_judge_min,
            zh.resolved_judge_input) == ("reranker-m3", "and", 0.6, 0.3, "heading")
    forced = Settings(embedding_model="m-en", data_dir=tmp_path, answerability_judge="nli-t",
                      answerability_input="section", _env_file=None)
    assert forced.resolved_judge == "nli-t" and forced.resolved_judge_input == "section"


def test_shipped_defaults_are_consistent():
    from rag.answerability import JUDGES, POLICIES
    from rag.answerer import JUDGE_INPUTS
    from rag.config import ANSWERABILITY_DEFAULTS
    for judge, policy, t_cos, t_judge, inp in ANSWERABILITY_DEFAULTS.values():
        assert judge in JUDGES and t_cos is not None
        if judge != "none":
            assert policy in POLICIES and t_judge is not None and inp in JUDGE_INPUTS


# ---------------------------------------------------------------- constrained selection
def _feat(i, typ, cos, j, correct=True):
    return {"id": str(i), "type": typ, "cos": cos, "rule": None, "correct": correct and typ != "unanswerable",
            "base_ms": 1.0, "j|heading": j, "j|heading_ms": 1.0, "j|plain": j, "j|plain_ms": 1.0,
            "j|section": j, "j|section_ms": 1.0}


def test_constrained_calibration_respects_false_refusal_cap():
    from rag.eval_m5 import calibrate_constrained, select
    # answerable 0..9, unanswerable 10..13; judge separates except answerable #0, #1 (low judge score)
    dev = [_feat(i, "factual", 0.8, 0.1 if i < 2 else 0.9) for i in range(10)] + \
          [_feat(10 + i, "unanswerable", 0.8, 0.05 if i < 2 else 0.5) for i in range(4)]
    v = {"name": "x", "policy": "replace", "judge": "j|heading", "rule": True, "input": "heading"}
    free = calibrate_constrained(dev, v, fr_cap=100.0)
    assert eval_ext.variant_metrics(dev, free)["false_refusal_rate"] == 20.0      # best bal. acc refuses #0,#1
    capped = calibrate_constrained(dev, v, fr_cap=10.0)
    m = eval_ext.variant_metrics(dev, capped)
    assert m["false_refusal_rate"] <= 10.0
    assert calibrate_constrained(dev, {**v, "policy": "cosine", "judge": None}, fr_cap=-1) is None
    sel = select(dev, ["j"], fr_cap=0.0)
    chosen = next(x for x in sel["variants"] if x["name"] == sel["chosen"])
    assert chosen["admissible"] and chosen["eligible"] and chosen["dev"]["false_refusal_rate"] == 0.0
    assert all(not x["eligible"] for x in sel["variants"] if x.get("input") == "plain")   # M4 input: ablation only


def test_fallback_rule():
    from rag.eval_m5 import fallback_rule
    s = lambda a, f: {"summary": {"answer_accuracy": a, "false_refusal_rate": f}}
    assert fallback_rule({"M3": s(66.7, 29.8), "M5": s(60.0, 20.0)})["triggered"]
    assert fallback_rule({"M3": s(66.7, 29.8), "M5": s(70.0, 31.0)})["triggered"]
    assert not fallback_rule({"M3": s(66.7, 29.8), "M5": s(66.7, 29.8)})["triggered"]


# ---------------------------------------------------------------- leakage guards
def test_similarity_measures():
    assert char_cosine("What does error code E204 mean?", "What does error code E205 mean?") > 0.8
    assert char_cosine("Who is the CEO?", "How do I bake sourdough bread?") < 0.3
    assert token_jaccard("How long is parental leave?", "parental leave: how long?") == 1.0
    new = [{"id": "n1", "question": "What does error code E205 mean?"},
           {"id": "n2", "question": "Who won the 2022 World Cup?"},
           {"id": "n3", "question": "who won the 2022 world cup"}]
    errs = check(new, [{"id": "t1", "question": "What does error code E204 mean?"}])
    assert any(e.startswith("n1:") for e in errs) and any("duplicate of n2" in e for e in errs)
    assert not any(e.startswith("n2:") for e in errs)


def test_test_split_is_frozen():
    sums = dict(reversed(x.split()) for x in (DS / "SHA256SUMS").read_text().splitlines() if x.strip())
    assert set(sums) == {"en.test.jsonl", "zh.test.jsonl"}
    for name, digest in sums.items():
        assert sha256(DS / name) == digest, f"{name} changed since M3/M4"


@pytest.mark.parametrize("lang,minimum", [("en", 40), ("zh", 15)])
def test_dev_extension_is_new_unanswerable_and_not_leaked(lang, minimum):
    extra = _rows(DS / f"{lang}.dev_extra.jsonl")
    test, dev = _rows(DS / f"{lang}.test.jsonl"), _rows(DS / f"{lang}.dev.jsonl")
    assert len(extra) >= minimum and all(r["type"] == "unanswerable" for r in extra)
    ids = [r["id"] for r in extra]
    assert len(set(ids)) == len(ids) and not set(ids) & {r["id"] for r in test + dev}
    assert check(extra, test + dev) == []
    # the jsonl is exactly what the authoring file + build script produce
    import importlib.util
    spec = importlib.util.spec_from_file_location("bde", ROOT / "scripts" / "build_dev_extra.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.build(lang) == extra
