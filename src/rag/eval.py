"""Offline evaluation: retrieval (hit@k, MRR), extractive/LLM answers, citation
correctness and refusal calibration.

    python -m rag.eval --lang en --model BAAI/bge-small-en-v1.5
    python -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5 --llm

Datasets are split into a frozen dev split (used to tune alpha and the refusal threshold)
and a held-out test split (headline numbers).
    python -m rag.eval --report          # rebuild eval/results/REPORT.md from all JSON results

Gold labels are (doc name, evidence substring) pairs, so they survive any change
of chunk size / overlap: a retrieved chunk is "gold" if it comes from the gold
document and contains the evidence text (whitespace/case-insensitive).
"""
from __future__ import annotations

import argparse
import os
import json
import re
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from . import eval_ext
from .answerer import ExtractiveAnswerer, OpenAICompatAnswerer
from .config import PROJECT_ROOT, Settings, apply_model_cache_env
from .service import RAGService
from .store import StoredChunk

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS = EVAL_DIR / "results"
KS = (1, 3, 5)
MODES = ("vector", "bm25", "hybrid")


# ----------------------------------------------------------------- helpers
def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def nospace(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def load_dataset(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_gold(chunk: StoredChunk, ev: dict) -> bool:
    return chunk.doc_name == ev["doc"] and nospace(ev["text"]) in nospace(chunk.text)


def answer_correct(answer: str, must: list[str]) -> bool:
    a = nospace(answer)
    return all(any(nospace(alt) in a for alt in m.split("|")) for m in must)


def validate_dataset(items: list[dict], docs_dir: Path) -> list[str]:
    """Every evidence string must occur in its document (catches typos in gold labels)."""
    from .ingest import parse_document
    texts = {p.name: nospace(" ".join(pg.text for pg in parse_document(p.name, p.read_bytes())))
             for p in docs_dir.iterdir() if p.is_file()}
    errs = []
    for it in items:
        for ev in it["evidence"]:
            if ev["doc"] not in texts:
                errs.append(f"{it['id']}: unknown doc {ev['doc']}")
            elif nospace(ev["text"]) not in texts[ev["doc"]]:
                errs.append(f"{it['id']}: evidence not found in {ev['doc']}: {ev['text']!r}")
    return errs


def pct(x: float) -> float:
    return round(100 * x, 1)


def auc(rows: list[dict], key: str) -> float:
    """ROC AUC of `key` separating answerable (positive) from unanswerable questions."""
    pos = [r[key] for r in rows if r["type"] != "unanswerable"]
    neg = [r[key] for r in rows if r["type"] == "unanswerable"]
    if not pos or not neg:
        return float("nan")
    return round(sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg)), 3)


# ----------------------------------------------------------------- retrieval
def eval_retrieval(svc: RAGService, items: list[dict], mode: str, depth: int = 10, expander=None) -> dict:
    rows, per_type = [], {}
    for it in items:
        if it["type"] == "unanswerable":
            continue
        extra = expander.expand(it["question"], svc.retriever) if expander is not None else None
        hits = svc.retriever.retrieve(it["question"], k=depth, mode=mode, expansions=extra,
                                      expansion_weight=expander.weight if expander is not None else 1.0)
        rank = next((r for r, h in enumerate(hits, 1) if any(is_gold(h.chunk, ev) for ev in it["evidence"])), None)
        covered5 = sum(any(is_gold(h.chunk, ev) for h in hits[:5]) for ev in it["evidence"]) / len(it["evidence"])
        rows.append({"id": it["id"], "type": it["type"], "rank": rank, "evidence_recall@5": covered5,
                     "expansion_ms": round(expander.last_cost_ms, 1) if expander is not None else 0.0,
                     "expansions": extra or []})
    def agg(rs):
        n = len(rs) or 1
        d = {f"hit@{k}": pct(sum(1 for r in rs if r["rank"] and r["rank"] <= k) / n) for k in KS}
        d["mrr"] = round(sum(1 / r["rank"] for r in rs if r["rank"]) / n, 3)
        d["evidence_recall@5"] = pct(sum(r["evidence_recall@5"] for r in rs) / n)
        d["n"] = len(rs)
        return d
    for t in sorted({r["type"] for r in rows}):
        per_type[t] = agg([r for r in rows if r["type"] == t])
    return {"overall": agg(rows), "by_type": per_type, "rows": rows}


# ----------------------------------------------------------------- answers
def run_answers(svc: RAGService, items: list[dict], answerer, top_k: int, expander=None) -> list[dict]:
    chunk_by_id = {c.chunk_id: c for c in svc.store.chunks}
    out = []
    for it in items:
        t0 = time.perf_counter()
        extra = expander.expand(it["question"], svc.retriever) if expander is not None else None
        hits = svc.retriever.retrieve(it["question"], k=top_k, mode="hybrid", expansions=extra,
                                      expansion_weight=expander.weight if expander is not None else 1.0)
        ans = answerer.answer(it["question"], hits)
        latency = (time.perf_counter() - t0) * 1000
        if expander is not None:  # count the LLM time even when the expansion came from the cache
            latency += expander.last_cost_ms - expander.last_ms
        cited = [chunk_by_id[c.chunk_id] for c in ans.citations]
        ev_cov = (sum(any(is_gold(ch, ev) for ch in cited) for ev in it["evidence"]) / len(it["evidence"])
                  if it["evidence"] else None)
        out.append({
            "id": it["id"], "type": it["type"], "question": it["question"], "answer": ans.answer,
            "refused": ans.refused, "refusal_reason": ans.refusal_reason, "confidence": ans.confidence,
            "backend": ans.backend,
            "citation_mode": ans.citation_mode, "invalid_markers": ans.invalid_markers,
            "citations": [f"{c.doc_name}{'' if c.page is None else f' p.{c.page}'}" for c in ans.citations],
            "answer_correct": (answer_correct(ans.answer, it["answer_contains"]) if it["answer_contains"] else None),
            "citation_correct": (ev_cov > 0 if ev_cov is not None and not ans.refused else None),
            "evidence_coverage": ev_cov, "latency_ms": round(latency, 1),
        })
    return out


def summarize_answers(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    answered = [r for r in ans if not r["refused"]]
    lat = sorted(r["latency_ms"] for r in rows)
    def frac(xs, cond):
        return pct(sum(1 for x in xs if cond(x)) / len(xs)) if xs else None
    return {
        "answer_accuracy": frac(ans, lambda r: r["answer_correct"] and not r["refused"]),
        "citation_accuracy": frac(answered, lambda r: r["citation_correct"]),
        "explicit_citation_rate": frac(answered, lambda r: r["citation_mode"] == "explicit"),
        "invalid_marker_rate": frac(answered, lambda r: bool(r["invalid_markers"])),
        "refusal_accuracy": frac(una, lambda r: r["refused"]),
        "false_refusal_rate": frac(ans, lambda r: r["refused"]),
        "latency_p50_ms": round(statistics.median(lat), 1) if lat else None,
        "latency_p95_ms": round(lat[min(len(lat) - 1, int(0.95 * len(lat)))], 1) if lat else None,
        "n_answerable": len(ans), "n_unanswerable": len(una),
    }


def effective_conf(r: dict) -> float:
    """Confidence as seen by the threshold; rows refused by a rule (no context, unknown
    identifier) are refused at any threshold, so they count as 0."""
    if r.get("refused") and r.get("refusal_reason") not in (None, "low_confidence"):
        return 0.0
    return r["confidence"] or 0.0


def calibrate(rows: list[dict]) -> dict:
    """Sweep the refusal threshold over extractive confidences (computed with min_score=0)."""
    ans = [effective_conf(r) for r in rows if r["type"] != "unanswerable"]
    una = [effective_conf(r) for r in rows if r["type"] == "unanswerable"]
    sweep = []
    for i in range(80, 181):
        t = i / 200
        ra = sum(c < t for c in una) / len(una)
        fr = sum(c < t for c in ans) / len(ans)
        sweep.append({"threshold": t, "refusal_accuracy": pct(ra), "false_refusal_rate": pct(fr),
                      "balanced_accuracy": pct((ra + 1 - fr) / 2)})
    best_bal = max(s["balanced_accuracy"] for s in sweep)
    plateau = [s for s in sweep if s["balanced_accuracy"] == best_bal]
    # midpoint of the best plateau = max margin to both classes
    best = plateau[len(plateau) // 2]
    return {"best": best, "sweep": sweep,
            "answerable_conf": {"min": round(min(ans), 3), "p10": round(sorted(ans)[len(ans) // 10], 3),
                                "median": round(statistics.median(ans), 3)},
            "unanswerable_conf": {"max": round(max(una), 3), "median": round(statistics.median(una), 3)}}


# ----------------------------------------------------------------- main
SPLITS = ("dev", "test")
ALPHAS = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def load_split(lang: str, split: str) -> list[dict]:
    return load_dataset(EVAL_DIR / "datasets" / f"{lang}.{split}.jsonl")


def tune_alpha(svc: RAGService, dev: list[dict], test: list[dict]) -> dict:
    """Pick the hybrid weight on dev (best MRR, ties -> closest to 0.5); test is reported only."""
    sweep = []
    for a in ALPHAS:
        svc.retriever.alpha = a
        sweep.append({"alpha": a, "dev_mrr": eval_retrieval(svc, dev, "hybrid")["overall"]["mrr"],
                      "test_mrr": eval_retrieval(svc, test, "hybrid")["overall"]["mrr"]})
    best = max(sweep, key=lambda x: (x["dev_mrr"], -abs(x["alpha"] - 0.5)))
    return {"chosen": best["alpha"], "sweep": sweep}


def threshold_metrics(rows: list[dict], t: float) -> dict:
    ans = [effective_conf(r) for r in rows if r["type"] != "unanswerable"]
    una = [effective_conf(r) for r in rows if r["type"] == "unanswerable"]
    ra, fr = sum(c < t for c in una) / len(una), sum(c < t for c in ans) / len(ans)
    return {"threshold": t, "refusal_accuracy": pct(ra), "false_refusal_rate": pct(fr),
            "balanced_accuracy": pct((ra + 1 - fr) / 2)}


def run(lang: str, model: str, chunk_sizes: list[int], llm: bool, llm_models: list[str], llm_url: str,
        top_k: int, model_dir: Path, llm_gating: tuple[bool, ...] = (True, False), judges: list[str] = (),
        qe_configs: list[dict] = ()) -> dict:
    data = {sp: load_split(lang, sp) for sp in SPLITS}
    docs_dir = PROJECT_ROOT / "sample_docs" / lang
    errs = [e for sp in SPLITS for e in validate_dataset(data[sp], docs_dir)]
    if errs:
        raise SystemExit("Dataset errors:\n" + "\n".join(errs))
    apply_model_cache_env(model_dir)
    result = {"lang": lang, "model": model, "top_k": top_k, "n_questions": {sp: len(data[sp]) for sp in SPLITS},
              "n_docs": len([p for p in docs_dir.iterdir() if p.is_file()]),
              "date": time.strftime("%Y-%m-%d %H:%M %Z"), "chunking": {}}
    embedder = None
    dev, test = data["dev"], data["test"]
    for cs in chunk_sizes:
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(data_dir=Path(tmp), model_dir=model_dir, embedding_backend="fastembed",
                         embedding_model=model, chunk_size=cs, chunk_overlap=max(cs // 7, 20), top_k=top_k,
                         answerability_judge="none", query_expansion="off", _env_file=None)
            svc = RAGService(s, embedder=embedder)
            embedder = svc.embedder
            t0 = time.perf_counter()
            for p in sorted(docs_dir.iterdir()):
                svc.ingest(p.name, p.read_bytes())
            entry = {"n_chunks": len(svc.store.chunks), "ingest_seconds": round(time.perf_counter() - t0, 2)}
            # 1) hybrid weight: tuned on dev only
            entry["alpha"] = tune_alpha(svc, dev, test)
            svc.retriever.alpha = entry["alpha"]["chosen"]
            entry["retrieval"] = {sp: {m: eval_retrieval(svc, data[sp], m) for m in MODES} for sp in SPLITS}
            # 2) refusal threshold: calibrated on dev only, then frozen and applied to test
            ext0 = ExtractiveAnswerer(embedder=embedder, min_score=0.0)
            rows0 = {sp: run_answers(svc, data[sp], ext0, top_k) for sp in SPLITS}
            cal = calibrate(rows0["dev"])
            thr = cal["best"]["threshold"]
            oracle = calibrate(rows0["test"])["best"]  # what we'd pick if we (illegally) tuned on test
            entry["calibration"] = {**cal, "dev": cal["best"], "test_at_dev_threshold": threshold_metrics(rows0["test"], thr),
                                    "test_oracle": oracle}
            # ablation: same protocol without the unknown-identifier rule
            abl0 = ExtractiveAnswerer(embedder=embedder, min_score=0.0, check_identifiers=False)
            abl_rows0 = {sp: run_answers(svc, data[sp], abl0, top_k) for sp in SPLITS}
            abl_thr = calibrate(abl_rows0["dev"])["best"]["threshold"]
            entry["ablation_no_identifier_rule"] = {"threshold": abl_thr, **{
                sp: threshold_metrics(abl_rows0[sp], abl_thr) for sp in SPLITS}}
            # M4 blocks (default chunk size only)
            if cs == chunk_sizes[0]:
                if judges:
                    from .answerability import build_judge
                    jm = {j: build_judge(j, model_dir) for j in judges}
                    feats = {sp: eval_ext.answerability_features(svc, data[sp], jm, top_k, embedder, answer_correct)
                             for sp in SPLITS}
                    entry["answerability"] = {**eval_ext.eval_answerability(feats, list(judges)), "features": feats}
                    print(f"   answerability: chosen on dev = {entry['answerability']['chosen']}")
                if qe_configs:
                    entry["query_expansion"] = eval_ext.eval_query_expansion(
                        svc, data, list(qe_configs), eval_retrieval, cache_path=RESULTS / f"qe_cache_{lang}.json")
                    print(f"   query expansion: chosen on dev = {entry['query_expansion']['chosen']}")
                entry["heading_context"] = eval_ext.eval_heading(svc, data, embedder, thr, top_k, run_answers,
                                                                 summarize_answers)
                if judges and qe_configs:  # end-to-end M4 system with the dev-chosen components
                    entry["m4_system"] = eval_ext.eval_system(
                        svc, data, embedder, entry, jm, top_k, run_answers, summarize_answers,
                        cache_path=RESULTS / f"qe_cache_{lang}.json")
                    for sp in SPLITS:
                        print(f"   {sp}: M4 system={entry['m4_system'][sp]['summary']}")
            ext = ExtractiveAnswerer(embedder=embedder, min_score=thr)
            entry["extractive"] = {"threshold": thr}
            for sp in SPLITS:
                rows = run_answers(svc, data[sp], ext, top_k)
                entry["extractive"][sp] = {"summary": summarize_answers(rows), "rows": rows}
            # 3) LLM: held-out test split only, default chunk size only (slow on CPU)
            entry["llm"] = []
            if llm and cs == chunk_sizes[0]:
                for llm_model in llm_models:
                    run_answers(svc, test[:1], OpenAICompatAnswerer(llm_url, llm_model, timeout=180), top_k)  # warm-up
                    for gated in llm_gating:
                        llm_ans = OpenAICompatAnswerer(llm_url, llm_model, timeout=180, fallback=ext,
                                                       gate=ext if gated else None)
                        lrows = run_answers(svc, test, llm_ans, top_k)
                        entry["llm"].append({"model": llm_model, "gated": gated, "threshold": thr if gated else None,
                                             "split": "test", "summary": summarize_answers(lrows), "rows": lrows})
            result["chunking"][str(cs)] = entry
            print(f"[{lang}] {model} chunk={cs}: chunks={entry['n_chunks']} alpha={entry['alpha']['chosen']} thr={thr}")
            for sp in SPLITS:
                print(f"   {sp}: hybrid={entry['retrieval'][sp]['hybrid']['overall']}")
                print(f"   {sp}: extractive={entry['extractive'][sp]['summary']}")
            for le in entry["llm"]:
                print(f"   LLM {le['model']} gated={le['gated']} (test): {le['summary']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{lang}__{model.replace('/', '_')}.json"
    if out.exists():  # keep previously recorded LLM results for (model, gating) pairs not re-run now
        old = json.loads(out.read_text())
        if isinstance(old.get("n_questions"), dict):  # only merge results produced on the same split format
            for cs, e in result["chunking"].items():
                done = {(x["model"], x["gated"]) for x in e["llm"]}
                prev = old.get("chunking", {}).get(cs, {})
                if prev.get("extractive", {}).get("threshold") == e["extractive"]["threshold"]:
                    e["llm"] += [x for x in prev.get("llm", []) if (x["model"], x["gated"]) not in done]
                for key in ("answerability", "query_expansion"):  # M4 blocks not re-run this time
                    if key not in e and key in prev:
                        e[key] = prev[key]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return result


def short(r: dict) -> str:
    return f"`{r['model'].split('/')[-1]}`"


def report() -> Path:
    results = [r for r in (json.loads(p.read_text()) for p in sorted(RESULTS.glob("*__*.json")))
               if isinstance(r, dict) and isinstance(r.get("n_questions"), dict)]
    L = ["# Evaluation report", "",
         "Generated by `make eval` (`python -m rag.eval`). Datasets: `eval/datasets/{lang}.{dev,test}.jsonl` "
         "(frozen, stratified by question type; built by `scripts/split_dataset.py`); corpora: `sample_docs/{lang}/`.",
         "",
         "**Protocol.** The hybrid weight (alpha) and the refusal threshold are tuned on **dev only** and then frozen. "
         "**test** is the held-out split and gives the headline numbers; dev is shown next to it so the dev→test gap "
         "(optimism from tuning) is visible. Retrieval metrics are computed on answerable questions; a retrieved chunk "
         "is a hit if it comes from the gold document and contains the gold evidence text. Answers use hybrid "
         "retrieval with top_k chunks.", ""]
    L += ["| lang | embedding | docs | dev questions | test questions |", "|---|---|---|---|---|"]
    for r in results:
        L.append(f"| {r['lang']} | `{r['model']}` | {r.get('n_docs', '–')} | {r['n_questions']['dev']} | {r['n_questions']['test']} |")
    L += ["", "## 1. Retrieval: vector vs BM25 vs hybrid (test, with dev in brackets)", "",
          "| lang | embedding | chunk | #chunks | mode | hit@1 | hit@3 | hit@5 | MRR | evidence recall@5 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            for m in MODES:
                t, d = e["retrieval"]["test"][m]["overall"], e["retrieval"]["dev"][m]["overall"]
                c = lambda k: f"{t[k]} ({d[k]})"
                L.append(f"| {r['lang']} | {short(r)} | {cs} | {e['n_chunks']} | {m} | {c('hit@1')} | {c('hit@3')} | {c('hit@5')} "
                         f"| {c('mrr')} | {c('evidence_recall@5')} |")
    L += ["", "### Hybrid weight (alpha = weight of the dense ranking in RRF), chosen on dev MRR", "",
          "| lang | embedding | chunk | " + " | ".join(f"α={a}" for a in ALPHAS) + " | chosen |",
          "|---|---|---|" + "---|" * (len(ALPHAS) + 1)]
    for r in results:
        for cs, e in r["chunking"].items():
            sw = {x["alpha"]: x for x in e["alpha"]["sweep"]}
            L.append(f"| {r['lang']} | {short(r)} | {cs} | " + " | ".join(
                f"{sw[a]['dev_mrr']} / {sw[a]['test_mrr']}" for a in ALPHAS) + f" | **{e['alpha']['chosen']}** |")
    L += ["", "Cells: dev MRR / test MRR.", "",
          "### Hybrid retrieval by question type, test (hit@1 / hit@3 / MRR)", "",
          "| lang | embedding | chunk | factual | paraphrase | cross-document |", "|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            bt = e["retrieval"]["test"]["hybrid"]["by_type"]
            cell = lambda t: (f"{bt[t]['hit@1']} / {bt[t]['hit@3']} / {bt[t]['mrr']} (n={bt[t]['n']})" if t in bt else "–")
            L.append(f"| {r['lang']} | {short(r)} | {cs} | {cell('factual')} | {cell('paraphrase')} | {cell('cross')} |")
    L += ["", "## 2. Answers: extractive vs local LLM", "",
          "answer acc = answerable questions answered (not refused) with the expected fact; citation acc = answered "
          "questions where at least one cited chunk is a gold evidence chunk; refusal acc = unanswerable questions "
          "correctly refused; false refusal = answerable questions wrongly refused. Latency is end-to-end per question "
          "(retrieval + answer) on CPU. The threshold is the dev-calibrated one. LLM runs are on test only.", "",
          "| lang | embedding | chunk | answerer | split | threshold | answer acc | citation acc | explicit [n] | invalid [n] "
          "| refusal acc | false refusal | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            runs = [("extractive", sp, e["extractive"]["threshold"], e["extractive"][sp]["summary"]) for sp in ("test", "dev")]
            runs += [(f"LLM `{x['model']}` {'+ gate' if x['gated'] else '(no gate)'}", x.get("split", "test"),
                      x.get("threshold"), x["summary"]) for x in e.get("llm", [])]
            if e.get("m4_system"):
                m4 = e["m4_system"]
                runs[2:2] = [(f"**M4 extractive**: {m4['answerability']} + {m4['query_expansion']} + headings", sp, None,
                              m4[sp]["summary"]) for sp in ("test", "dev")]
            for name, sp, thr, s in runs:
                L.append(f"| {r['lang']} | {short(r)} | {cs} | {name} | {sp} | {thr or '–'} "
                         f"| {s['answer_accuracy']} | {s['citation_accuracy']} | {s['explicit_citation_rate']} "
                         f"| {s['invalid_marker_rate']} | {s['refusal_accuracy']} | {s['false_refusal_rate']} "
                         f"| {s['latency_p50_ms']} | {s['latency_p95_ms']} |")
    L += ["", "## 3. Refusal threshold calibration (extractive confidence)", "",
          "Confidence = max cosine(sentence, question) over sentences of the retrieved chunks. The threshold is the "
          "midpoint of the best balanced-accuracy plateau **on dev**; it is then applied unchanged to test. "
          "\"test oracle\" is what tuning on test would have picked — the gap between the two rows is the optimism "
          "you would report if you calibrated on the evaluation set.", "",
          "| lang | embedding | chunk | dev threshold | dev refusal / false-refusal / bal. | test @ dev threshold | test-oracle threshold | test-oracle bal. |",
          "|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            c = e["calibration"]
            d, t, o = c["dev"], c["test_at_dev_threshold"], c["test_oracle"]
            L.append(f"| {r['lang']} | {short(r)} | {cs} | **{d['threshold']}** | {d['refusal_accuracy']} / {d['false_refusal_rate']} / "
                     f"{d['balanced_accuracy']} | {t['refusal_accuracy']} / {t['false_refusal_rate']} / {t['balanced_accuracy']} "
                     f"| {o['threshold']} | {o['balanced_accuracy']} |")
    L += ["", "Ablation — same dev-calibration protocol without the unknown-identifier refusal rule "
          "(refusal acc / false refusal / balanced acc):", "",
          "| lang | embedding | chunk | threshold | dev | test |", "|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            a = e.get("ablation_no_identifier_rule")
            if a:
                f = lambda m: f"{m['refusal_accuracy']} / {m['false_refusal_rate']} / {m['balanced_accuracy']}"
                L.append(f"| {r['lang']} | {short(r)} | {cs} | {a['threshold']} | {f(a['dev'])} | {f(a['test'])} |")
    L += ["", "## 4. Failures (extractive, test split, first chunk size per run)", ""]
    for r in results:
        cs, e = next(iter(r["chunking"].items()))
        rows = e["extractive"]["test"]["rows"]
        bad = [x for x in rows if (x["type"] == "unanswerable") != x["refused"] or (x["answer_correct"] is False)]
        L.append(f"**{r['lang']} / `{r['model']}` / chunk {cs}** — {len(bad)} of {len(rows)}")
        L.append("")
        for x in bad:
            L.append(f"- `{x['id']}` ({x['type']}, conf {x['confidence']}): {x['question']} → "
                     f"{x['answer'][:140].replace('|', '/')}{'…' if len(x['answer']) > 140 else ''}")
        L.append("")
    L += m4_report(results)
    from .eval_m5 import report_lines
    L += report_lines()
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    return out


AUC_KEYS = ("cos", "nli", "nli-t", "reranker", "reranker-m3")


def m4_report(results: list[dict]) -> list[str]:
    L = ["## 5. Answerability: cosine threshold vs NLI / cross-encoder judges (M4)", "",
         "Each variant's threshold(s) are chosen on **dev** (max balanced accuracy, middle of the plateau; 2-D grid "
         "for AND/OR) and applied unchanged to **test**. Judges score the top-3 candidate sentences: `nli` = "
         "mDeBERTa-v3 XNLI fp32 ONNX, max P(entailment) with the question as hypothesis; `nli-t` = same with "
         "\"The text answers the question: …\"; `reranker` = ms-marco-MiniLM-L-12 (en); `reranker-m3` = "
         "bge-reranker-v2-m3 uint8 (multilingual). answer acc = answerable questions answered with the expected fact "
         "(threshold-free answer text, so variants differ only by what they refuse). Latency p50 = retrieval + "
         "extractive answer + judge, CPU. **Bold** = variant chosen on dev.", "",
         "| lang | variant | thresholds (cos / judge) | dev bal. acc | test refusal acc | test false refusal | test bal. acc "
         "| test answer acc | judge ms p50 | latency p50 ms |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        e = next(iter(r["chunking"].values()))
        a = e.get("answerability")
        if not a:
            continue
        for v in a["variants"]:
            nm = f"**{v['name']}**" if v["name"] == a["chosen"] else v["name"]
            th = f"{v.get('t_cos') if v.get('t_cos') is not None else '–'} / {v.get('t_judge') if v.get('t_judge') is not None else '–'}"
            d, t = v["dev"], v["test"]
            L.append(f"| {r['lang']} | {nm} | {th} | {d['balanced_accuracy']} | {t['refusal_accuracy']} | "
                     f"{t['false_refusal_rate']} | {t['balanced_accuracy']} | {t['answer_accuracy']} | {t['judge_ms_p50']} "
                     f"| {t['latency_p50_ms']} |")
    L += ["", "ROC AUC (answerable vs unanswerable) of each raw score — dev / test:", "",
          "| lang | " + " | ".join(AUC_KEYS) + " |", "|---" * (len(AUC_KEYS) + 1) + "|"]
    for r in results:
        a = next(iter(r["chunking"].values())).get("answerability")
        if a:
            L.append(f"| {r['lang']} | " + " | ".join(
                f"{a['auc']['dev'][k]} / {a['auc']['test'][k]}" if k in a["auc"]["dev"] else "–"
                for k in AUC_KEYS) + " |")
    L += ["", "## 6. Query expansion for paraphrased questions (M4)", "",
          "Hybrid retrieval (dev-tuned alpha); expansions are fused with the original query by RRF, expansion weight "
          "tuned on dev. LLM expansions use local Ollama (temperature 0); expansion latency is the extra time per "
          "question on CPU (cached in `eval/results/qe_cache_*.json`, measured when first computed). **Bold** = default "
          "chosen on dev MRR (all answerable questions).", "",
          "| lang | expansion | weight | dev MRR | test hit@1 | test MRR | test paraphrase hit@1 | test paraphrase MRR "
          "| extra ms p50 | extra ms p95 |", "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        q = next(iter(r["chunking"].values())).get("query_expansion")
        if not q:
            continue
        for c in q["configs"]:
            nm = f"**{c['name']}**" if c["name"] == q["chosen"] else c["name"]
            t, para = c["test"]["overall"], c["test"]["by_type"].get("paraphrase", {})
            L.append(f"| {r['lang']} | {nm} | {c['weight'] or '–'} | {c['dev']['overall']['mrr']} | {t['hit@1']} | {t['mrr']} "
                     f"| {para.get('hit@1', '–')} | {para.get('mrr', '–')} | {c['test'].get('expansion_ms_p50', 0)} "
                     f"| {c['test'].get('expansion_ms_p95', 0)} |")
    L += ["", "## 7. Heading context in extractive answers (M4)", "",
          "Answer accuracy of the extractive answerer (dev-calibrated threshold) without / with the section heading "
          "prefixed when it carries the asked-for date, version or terms (`HEADING_CONTEXT`).", "",
          "| lang | split | answer acc off | answer acc on | answers changed |", "|---|---|---|---|---|"]
    ex = []
    for r in results:
        h = next(iter(r["chunking"].values())).get("heading_context")
        if not h:
            continue
        for sp in SPLITS:
            L.append(f"| {r['lang']} | {sp} | {h[sp]['off']} | {h[sp]['on']} | {h[sp]['n_changed']} |")
            ex += [(r["lang"], sp, c) for c in h[sp]["changed"]]
    if ex:
        L += ["", "Changed answers:", ""]
        for lang, sp, c in ex:
            mark = {(False, True): "fixed", (True, False): "broken"}.get((bool(c["correct_off"]), bool(c["correct_on"])), "same")
            L.append(f"- `{c['id']}` ({lang} {sp}, {mark}): {c['question']} → {c['on'][:200].replace('|', '/')}")
    return L + [""]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "zh"])
    ap.add_argument("--model")
    ap.add_argument("--chunk-sizes", default="800")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--llm", action="store_true", help="also evaluate the OpenAI-compatible LLM backend")
    ap.add_argument("--llm-models", default="qwen2.5:0.5b", help="comma-separated")
    ap.add_argument("--llm-gating", default="gated,ungated", help="gated, ungated or both")
    ap.add_argument("--llm-url", default="http://127.0.0.1:11434/v1")
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models")))
    ap.add_argument("--report", action="store_true", help="only rebuild REPORT.md")
    ap.add_argument("--judges", default="", help="answerability judges to compare, e.g. nli,reranker,reranker-m3")
    ap.add_argument("--qe", default="", help="query-expansion configs, e.g. prf,rewrite:qwen2.5:1.5b,hyde:qwen2.5:0.5b")
    a = ap.parse_args(argv)
    if not a.report:
        if not (a.lang and a.model):
            ap.error("--lang and --model are required unless --report")
        gating = tuple(g == "gated" for g in a.llm_gating.split(","))
        qe = [{"mode": c.split(":", 1)[0], "model": c.split(":", 1)[1] if ":" in c else None, "url": a.llm_url}
              for c in a.qe.split(",") if c]
        run(a.lang, a.model, [int(x) for x in a.chunk_sizes.split(",")], a.llm, a.llm_models.split(","), a.llm_url,
            a.top_k, Path(a.model_dir), gating, [j for j in a.judges.split(",") if j], qe)
    report()


if __name__ == "__main__":
    main()
