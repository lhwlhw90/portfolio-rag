from pathlib import Path

import pytest

import importlib.util

from rag.eval import (EVAL_DIR, answer_correct, calibrate, effective_conf, eval_retrieval, is_gold, load_split,
                      summarize_answers, threshold_metrics, validate_dataset)
from rag.store import StoredChunk

ROOT = Path(__file__).resolve().parents[1]


TYPES = {"factual", "paraphrase", "cross", "unanswerable"}


@pytest.mark.parametrize("lang", ["en", "zh"])
@pytest.mark.parametrize("split", ["dev", "test"])
def test_datasets_are_valid(lang, split):
    items = load_split(lang, split)
    assert validate_dataset(items, ROOT / "sample_docs" / lang) == []
    assert {i["type"] for i in items} == TYPES
    assert len({i["id"] for i in items}) == len(items)
    for i in items:
        assert (i["type"] == "unanswerable") == (not i["evidence"])


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_splits_are_disjoint_large_and_stratified(lang):
    dev, test = load_split(lang, "dev"), load_split(lang, "test")
    assert len(dev) + len(test) >= 100
    assert not ({i["id"] for i in dev} & {i["id"] for i in test})
    assert not ({i["question"] for i in dev} & {i["question"] for i in test})
    for t in TYPES:  # every type is represented in both splits, roughly 40/60
        nd, nt = sum(i["type"] == t for i in dev), sum(i["type"] == t for i in test)
        assert nd >= 4 and nt >= 6 and 0.3 <= nd / (nd + nt) <= 0.5


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_committed_splits_match_authoring_source(lang):
    """The frozen split files must be exactly what scripts/split_dataset.py produces."""
    spec = importlib.util.spec_from_file_location("split_dataset", ROOT / "scripts" / "split_dataset.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    dev, test = mod.split(mod.load(lang))
    assert dev == load_split(lang, "dev") and test == load_split(lang, "test")


def test_zh_dataset_covers_chinese_pdf_pages():
    items = load_split("zh", "dev") + load_split("zh", "test")
    assert sum(any(e["doc"].endswith(".pdf") for e in i["evidence"]) for i in items) >= 10


def test_rule_refusals_count_as_refused_at_any_threshold():
    rows = [{"type": "factual", "confidence": 0.8, "refused": False, "refusal_reason": None},
            {"type": "unanswerable", "confidence": 0.9, "refused": True, "refusal_reason": "unknown_identifier"},
            {"type": "unanswerable", "confidence": 0.5, "refused": False, "refusal_reason": None}]
    assert effective_conf(rows[1]) == 0.0
    m = threshold_metrics(rows, 0.6)
    assert m["refusal_accuracy"] == 100.0 and m["false_refusal_rate"] == 0.0


def test_gold_matching_ignores_whitespace_and_case():
    ch = StoredChunk("d:0", "d", "a.md", 0, None, "Unused vacation days can be carried over\nto the next YEAR.")
    assert is_gold(ch, {"doc": "a.md", "text": "carried over to the next year"})
    assert not is_gold(ch, {"doc": "b.md", "text": "carried over"})


def test_answer_correct_all_groups_with_alternatives():
    assert answer_correct("It is 3-year, released 2025-07-01", ["3-year", "2025|2026"])
    assert not answer_correct("It is 3-year", ["3-year", "2025"])


def test_calibrate_finds_separating_threshold():
    rows = [{"type": "factual", "confidence": c} for c in (0.7, 0.75, 0.8)] + \
           [{"type": "unanswerable", "confidence": c} for c in (0.5, 0.55)]
    best = calibrate(rows)["best"]
    assert 0.55 < best["threshold"] <= 0.7 and best["balanced_accuracy"] == 100.0


def test_summarize_answers():
    rows = [{"type": "factual", "refused": False, "answer_correct": True, "citation_correct": True,
             "citation_mode": "explicit", "invalid_markers": [], "latency_ms": 10},
            {"type": "factual", "refused": True, "answer_correct": False, "citation_correct": None,
             "citation_mode": "explicit", "invalid_markers": [], "latency_ms": 20},
            {"type": "unanswerable", "refused": True, "answer_correct": None, "citation_correct": None,
             "citation_mode": "explicit", "invalid_markers": [], "latency_ms": 30}]
    s = summarize_answers(rows)
    assert s["answer_accuracy"] == 50.0 and s["false_refusal_rate"] == 50.0 and s["refusal_accuracy"] == 100.0
    assert s["citation_accuracy"] == 100.0


def test_eval_retrieval_smoke(loaded_service):
    items = [{"id": "x", "type": "factual", "question": "maximum payload RoboArm",
              "evidence": [{"doc": "product_faq.txt", "text": "maximum payload of 7.5 kg"}], "answer_contains": ["7.5"]}]
    for mode in ("vector", "bm25", "hybrid"):
        r = eval_retrieval(loaded_service, items, mode)
        assert r["overall"]["hit@5"] == 100.0 and r["overall"]["n"] == 1
