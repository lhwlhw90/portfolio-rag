from pathlib import Path

import pytest

from rag.eval import (EVAL_DIR, answer_correct, calibrate, eval_retrieval, is_gold, load_dataset,
                      summarize_answers, validate_dataset)
from rag.store import StoredChunk

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("lang", ["en", "zh"])
def test_datasets_are_valid(lang):
    items = load_dataset(EVAL_DIR / "datasets" / f"{lang}.jsonl")
    assert validate_dataset(items, ROOT / "sample_docs" / lang) == []
    types = {i["type"] for i in items}
    assert types == {"factual", "paraphrase", "cross", "unanswerable"}
    assert len({i["id"] for i in items}) == len(items)
    for i in items:
        assert (i["type"] == "unanswerable") == (not i["evidence"])


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
