"""Offline evaluation: retrieval (hit@k, MRR), extractive/LLM answers, citation
correctness and refusal calibration.

    python -m rag.eval --lang en --model BAAI/bge-small-en-v1.5
    python -m rag.eval --lang zh --model BAAI/bge-small-zh-v1.5 --llm
    python -m rag.eval --report          # rebuild eval/results/REPORT.md from all JSON results

Gold labels are (doc name, evidence substring) pairs, so they survive any change
of chunk size / overlap: a retrieved chunk is "gold" if it comes from the gold
document and contains the evidence text (whitespace/case-insensitive).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

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


# ----------------------------------------------------------------- retrieval
def eval_retrieval(svc: RAGService, items: list[dict], mode: str, depth: int = 10) -> dict:
    rows, per_type = [], {}
    for it in items:
        if it["type"] == "unanswerable":
            continue
        hits = svc.retriever.retrieve(it["question"], k=depth, mode=mode)
        rank = next((r for r, h in enumerate(hits, 1) if any(is_gold(h.chunk, ev) for ev in it["evidence"])), None)
        covered5 = sum(any(is_gold(h.chunk, ev) for h in hits[:5]) for ev in it["evidence"]) / len(it["evidence"])
        rows.append({"id": it["id"], "type": it["type"], "rank": rank, "evidence_recall@5": covered5})
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
def run_answers(svc: RAGService, items: list[dict], answerer, top_k: int) -> list[dict]:
    chunk_by_id = {c.chunk_id: c for c in svc.store.chunks}
    out = []
    for it in items:
        t0 = time.perf_counter()
        hits = svc.retriever.retrieve(it["question"], k=top_k, mode="hybrid")
        ans = answerer.answer(it["question"], hits)
        latency = (time.perf_counter() - t0) * 1000
        cited = [chunk_by_id[c.chunk_id] for c in ans.citations]
        ev_cov = (sum(any(is_gold(ch, ev) for ch in cited) for ev in it["evidence"]) / len(it["evidence"])
                  if it["evidence"] else None)
        out.append({
            "id": it["id"], "type": it["type"], "question": it["question"], "answer": ans.answer,
            "refused": ans.refused, "confidence": ans.confidence, "backend": ans.backend,
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


def calibrate(rows: list[dict]) -> dict:
    """Sweep the refusal threshold over extractive confidences (computed with min_score=0)."""
    ans = [r["confidence"] or 0.0 for r in rows if r["type"] != "unanswerable"]
    una = [r["confidence"] or 0.0 for r in rows if r["type"] == "unanswerable"]
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
def run(lang: str, model: str, chunk_sizes: list[int], llm: bool, llm_models: list[str], llm_url: str,
        top_k: int, model_dir: Path, llm_gating: tuple[bool, ...] = (True, False)) -> dict:
    items = load_dataset(EVAL_DIR / "datasets" / f"{lang}.jsonl")
    docs_dir = PROJECT_ROOT / "sample_docs" / lang
    errs = validate_dataset(items, docs_dir)
    if errs:
        raise SystemExit("Dataset errors:\n" + "\n".join(errs))
    apply_model_cache_env(model_dir)
    result = {"lang": lang, "model": model, "top_k": top_k, "n_questions": len(items),
              "date": time.strftime("%Y-%m-%d %H:%M %Z"), "chunking": {}}
    embedder = None
    for cs in chunk_sizes:
        with tempfile.TemporaryDirectory() as tmp:
            s = Settings(data_dir=Path(tmp), model_dir=model_dir, embedding_backend="fastembed",
                         embedding_model=model, chunk_size=cs, chunk_overlap=max(cs // 7, 20), top_k=top_k,
                         _env_file=None)
            svc = RAGService(s, embedder=embedder)
            embedder = svc.embedder
            t0 = time.perf_counter()
            for p in sorted(docs_dir.iterdir()):
                svc.ingest(p.name, p.read_bytes())
            ingest_s = time.perf_counter() - t0
            entry = {"n_chunks": len(svc.store.chunks), "ingest_seconds": round(ingest_s, 2),
                     "retrieval": {m: eval_retrieval(svc, items, m) for m in MODES}}
            ext0 = ExtractiveAnswerer(embedder=embedder, min_score=0.0)
            rows0 = run_answers(svc, items, ext0, top_k)
            cal = calibrate(rows0)
            entry["calibration"] = cal
            thr = cal["best"]["threshold"]
            ext = ExtractiveAnswerer(embedder=embedder, min_score=thr)
            rows = run_answers(svc, items, ext, top_k)
            entry["extractive"] = {"threshold": thr, "summary": summarize_answers(rows), "rows": rows}
            entry["llm"] = []
            if llm and cs == chunk_sizes[0]:  # LLM runs are slow: default chunk size only
                for llm_model in llm_models:
                    for gated in llm_gating:
                        llm_ans = OpenAICompatAnswerer(llm_url, llm_model, timeout=180, fallback=ext,
                                                       gate=ext if gated else None)
                        run_answers(svc, items[:1], OpenAICompatAnswerer(llm_url, llm_model, timeout=180), top_k)  # warm-up
                        lrows = run_answers(svc, items, llm_ans, top_k)
                        entry["llm"].append({"model": llm_model, "gated": gated, "threshold": thr if gated else None,
                                             "summary": summarize_answers(lrows), "rows": lrows})
            result["chunking"][str(cs)] = entry
            print(f"[{lang}] {model} chunk={cs}: chunks={entry['n_chunks']} "
                  f"hybrid={entry['retrieval']['hybrid']['overall']} thr={thr} "
                  f"extractive={entry['extractive']['summary']}")
            for le in entry["llm"]:
                print(f"   LLM {le['model']} gated={le['gated']}: {le['summary']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{lang}__{model.replace('/', '_')}.json"
    if out.exists():  # keep previously recorded LLM results for (model, gating) pairs not re-run now
        old = json.loads(out.read_text())
        for cs, e in result["chunking"].items():
            done = {(x["model"], x["gated"]) for x in e["llm"]}
            e["llm"] += [x for x in old.get("chunking", {}).get(cs, {}).get("llm", [])
                         if (x["model"], x["gated"]) not in done]
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {out}")
    return result


def _fmt(v):
    return "–" if v is None else (f"{v:.3f}" if isinstance(v, float) and v < 1.0001 and v != 0 and not float(v).is_integer() and v < 1 else str(v))


def report() -> Path:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS.glob("*.json"))]
    L = ["# Evaluation report", "",
         "Generated by `make eval` (`python -m rag.eval`). Datasets: `eval/datasets/{en,zh}.jsonl`; corpora: "
         "`sample_docs/{en,zh}/`. Retrieval metrics are computed on answerable questions; a retrieved chunk is a "
         "hit if it comes from the gold document and contains the gold evidence text. Answer metrics use hybrid "
         "retrieval with top_k chunks.", ""]
    L += ["## 1. Retrieval: vector vs BM25 vs hybrid", "",
          "| lang | embedding model | chunk | #chunks | mode | hit@1 | hit@3 | hit@5 | MRR | evidence recall@5 |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            for m in MODES:
                o = e["retrieval"][m]["overall"]
                L.append(f"| {r['lang']} | `{r['model']}` | {cs} | {e['n_chunks']} | {m} | {o['hit@1']} | {o['hit@3']} "
                         f"| {o['hit@5']} | {o['mrr']} | {o['evidence_recall@5']} |")
    L += ["", "### Hybrid retrieval by question type (hit@1 / hit@3 / MRR)", "",
          "| lang | model | chunk | factual | paraphrase | cross-document |", "|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            bt = e["retrieval"]["hybrid"]["by_type"]
            cell = lambda t: (f"{bt[t]['hit@1']} / {bt[t]['hit@3']} / {bt[t]['mrr']} (n={bt[t]['n']})" if t in bt else "–")
            L.append(f"| {r['lang']} | `{r['model']}` | {cs} | {cell('factual')} | {cell('paraphrase')} | {cell('cross')} |")
    L += ["", "## 2. Answers: extractive vs local LLM", "",
          "answer acc = answerable questions answered (not refused) with the expected fact; citation acc = answered "
          "questions where at least one cited chunk is a gold evidence chunk; refusal acc = unanswerable questions "
          "correctly refused; false refusal = answerable questions wrongly refused. Latency is end-to-end per question "
          "(retrieval + answer) on CPU.", "",
          "| lang | embedding | chunk | answerer | threshold | answer acc | citation acc | explicit [n] | invalid [n] "
          "| refusal acc | false refusal | p50 ms | p95 ms |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            runs = [("extractive", e["extractive"])] + [
                (f"LLM `{x['model']}` {'+ gate' if x['gated'] else '(no gate)'}", x) for x in e.get("llm", [])]
            for name, run_ in runs:
                s = run_["summary"]
                L.append(f"| {r['lang']} | `{r['model']}` | {cs} | {name} | {run_.get('threshold') or '–'} "
                         f"| {s['answer_accuracy']} | {s['citation_accuracy']} | {s['explicit_citation_rate']} "
                         f"| {s['invalid_marker_rate']} | {s['refusal_accuracy']} | {s['false_refusal_rate']} "
                         f"| {s['latency_p50_ms']} | {s['latency_p95_ms']} |")
    L += ["", "## 3. Refusal threshold calibration (extractive confidence)", "",
          "Confidence = max cosine(sentence, question) over sentences of the retrieved chunks. "
          "The chosen threshold is the midpoint of the plateau with the best balanced accuracy.", "",
          "| lang | model | chunk | answerable conf (min / p10 / median) | unanswerable conf (median / max) "
          "| chosen threshold | refusal acc | false refusal | balanced acc |", "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        for cs, e in r["chunking"].items():
            c = e["calibration"]
            a, u, b = c["answerable_conf"], c["unanswerable_conf"], c["best"]
            L.append(f"| {r['lang']} | `{r['model']}` | {cs} | {a['min']} / {a['p10']} / {a['median']} | {u['median']} / {u['max']} "
                     f"| **{b['threshold']}** | {b['refusal_accuracy']} | {b['false_refusal_rate']} | {b['balanced_accuracy']} |")
    L += ["", "## 4. Failures (extractive, first chunk size per run)", ""]
    for r in results:
        cs, e = next(iter(r["chunking"].items()))
        bad = [x for x in e["extractive"]["rows"] if (x["type"] == "unanswerable") != x["refused"]
               or (x["answer_correct"] is False)]
        L.append(f"**{r['lang']} / `{r['model']}` / chunk {cs}** — {len(bad)} of {len(e['extractive']['rows'])}")
        L.append("")
        for x in bad:
            L.append(f"- `{x['id']}` ({x['type']}, conf {x['confidence']}): {x['question']} → "
                     f"{x['answer'][:140].replace('|', '/')}{'…' if len(x['answer']) > 140 else ''}")
        L.append("")
    out = RESULTS / "REPORT.md"
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"wrote {out}")
    return out


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
    ap.add_argument("--model-dir", default=str(PROJECT_ROOT / "models"))
    ap.add_argument("--report", action="store_true", help="only rebuild REPORT.md")
    a = ap.parse_args(argv)
    if not a.report:
        if not (a.lang and a.model):
            ap.error("--lang and --model are required unless --report")
        gating = tuple(g == "gated" for g in a.llm_gating.split(","))
        run(a.lang, a.model, [int(x) for x in a.chunk_sizes.split(",")], a.llm, a.llm_models.split(","), a.llm_url,
            a.top_k, Path(a.model_dir), gating)
    report()


if __name__ == "__main__":
    main()
