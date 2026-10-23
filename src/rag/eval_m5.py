"""M5: re-select the answerability judge on an extended dev set under a false-refusal constraint.

Protocol (fixed before the single test evaluation, see docs/DESIGN.md section 13):
1. dev = the frozen M3/M4 dev split + eval/datasets/{lang}.dev_extra.jsonl (new unanswerable questions,
   leakage-checked against test by scripts/build_dev_extra.py). The test files must match SHA256SUMS.
2. Retrieval = the deployed pipeline: chunk 800, dev-tuned hybrid alpha (from the M3/M4 results JSON),
   the M4 dev-chosen query expansion, heading context on.
3. Candidates: cosine-only (re-calibrated) and every judge x policy (replace/and/or) x judge input
   ("heading" = the sentence as shown in the answer, heading prefixed by the with_heading rule;
   "section" = always prefix the section heading). "plain" (M4) input is reported as an ablation only.
4. Thresholds: grid on dev; only points whose dev false-refusal rate <= the M3 system's dev
   false-refusal rate are admissible; among them max dev balanced accuracy (plateau midpoint).
   The default is the admissible variant with the highest dev balanced accuracy (ties -> simplest).
5. `--final` then evaluates test ONCE (M3 / M4 / M5 / M5-fallback systems end-to-end) and applies the
   pre-registered fallback rule: if the English M5 system is worse than M3 on test (answer accuracy
   lower, or false-refusal rate higher), the English default becomes ANSWERABILITY_JUDGE=none.
   A second --final run is refused unless --force-retest is given (and then counted in the JSON).

    python -m rag.eval_m5 --lang en            # dev only (never reads test)
    python -m rag.eval_m5 --lang en --final    # one-time test evaluation
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path

from . import eval_ext
from .answerability import TimedJudge, build_judge
from .answerer import ExtractiveAnswerer, judge_text
from .config import (M4_ANSWERABILITY_DEFAULTS, MODEL_MIN_SCORES, PROJECT_ROOT, QUERY_EXPANSION_DEFAULTS, Settings,
                     apply_model_cache_env)
from .eval import (RESULTS, answer_correct, load_dataset, run_answers, summarize_answers, validate_dataset)
from .leakage import sha256
from .query import QueryExpander
from .service import RAGService
from .text import sentences_with_headings

DS = PROJECT_ROOT / "eval" / "datasets"
MODELS = {"en": "BAAI/bge-small-en-v1.5", "zh": "BAAI/bge-small-zh-v1.5"}
JUDGES = {"en": ["nli", "nli-t", "reranker", "reranker-m3"], "zh": ["nli", "nli-t", "reranker-m3"]}
INPUTS = ("plain", "heading", "section")
ELIGIBLE_INPUTS = ("heading", "section")
CHUNK, TOP_K = 800, 4


# ------------------------------------------------------------------ data
def verify_test_frozen() -> dict[str, str]:
    sums = dict(reversed(x.split()) for x in (DS / "SHA256SUMS").read_text().splitlines() if x.strip())
    for name, digest in sums.items():
        if sha256(DS / name) != digest:
            raise SystemExit(f"{name} differs from the frozen M3/M4 test set (SHA256SUMS)")
    return sums


def load_dev(lang: str) -> list[dict]:
    return load_dataset(DS / f"{lang}.dev.jsonl") + load_dataset(DS / f"{lang}.dev_extra.jsonl")


# ------------------------------------------------------------------ features
def features(svc, items, ext0, judges: dict, expander, inputs=INPUTS) -> list[dict]:
    """Per question: cosine, rule refusal, correctness of the threshold-free answer, and every
    judge's score for every judge input (identical inputs are scored once)."""
    timed = {n: TimedJudge(j) for n, j in judges.items()}
    rows = []
    for it in items:
        q = it["question"]
        t0 = time.perf_counter()
        extra = expander.expand(q, svc.retriever) if expander else None
        if expander is not None and expander._down_until:
            raise SystemExit(f"query expansion LLM failed on {it['id']}: start Ollama (./scripts/ollama.sh up)")
        hits = svc.retriever.retrieve(q, k=TOP_K, mode="hybrid", expansions=extra,
                                      expansion_weight=expander.weight if expander else 1.0)
        ans = ext0.answer(q, hits)
        base_ms = (time.perf_counter() - t0) * 1000 + ((expander.last_cost_ms - expander.last_ms) if expander else 0)
        cands = ext0.score_sentences(q, hits)[:3]
        heads = {r: sentences_with_headings(hits[r].chunk.text, getattr(hits[r].chunk, "heading", None))
                 for r in {c[1] for c in cands}}
        f = {"id": it["id"], "type": it["type"], "cos": ans.confidence or 0.0, "base_ms": round(base_ms, 1),
             "rule": ans.refusal_reason if ans.refused and ans.refusal_reason in eval_ext.RULE_REASONS else None,
             "correct": bool(it["answer_contains"]) and answer_correct(ans.answer, it["answer_contains"]),
             "texts": {}}
        for inp in inputs:
            f["texts"][inp] = [judge_text(c[3], heads[c[1]][c[2]][1], q, inp) for c in cands]
        for n, j in timed.items():
            memo: dict[tuple, tuple[float, float]] = {}
            for inp in inputs:
                key = tuple(f["texts"][inp])
                if key not in memo:
                    memo[key] = ((round(j.score(q, list(key)), 4), round(j.times_ms[-1], 1)) if key else (0.0, 0.0))
                f[f"{n}|{inp}"], f[f"{n}|{inp}_ms"] = memo[key]
        rows.append(f)
    return rows


# ------------------------------------------------------------------ selection
def calibrate_constrained(dev: list[dict], v: dict, fr_cap: float) -> dict | None:
    """Thresholds maximizing dev balanced accuracy subject to dev false refusal <= fr_cap
    (plateau midpoint). None if no threshold satisfies the constraint."""
    cos_grid = [t / 200 for t in range(80, 181)] if v["policy"] != "replace" else [None]
    j_grid = eval_ext._grid([r[v["judge"]] for r in dev]) if v.get("judge") else [None]
    ok = []
    for tc in cos_grid:
        for tj in j_grid:
            m = eval_ext.variant_metrics(dev, {**v, "t_cos": tc, "t_judge": tj})
            if m["false_refusal_rate"] <= fr_cap + 1e-9:
                ok.append((m["balanced_accuracy"], tc, tj))
    if not ok:
        return None
    best = max(s[0] for s in ok)
    plateau = [s for s in ok if s[0] == best]
    _, tc, tj = plateau[len(plateau) // 2]
    return {**v, "t_cos": tc, "t_judge": tj}


def variants(judge_names: list[str]) -> list[dict]:
    out = [{"name": "cosine + identifier rule (re-calibrated)", "policy": "cosine", "rule": True, "input": None}]
    for j in judge_names:
        for inp in INPUTS:
            for pol, label in (("replace", "only"), ("and", "AND cosine"), ("or", "OR cosine")):
                out.append({"name": f"{j}[{inp}] {label}", "policy": pol, "judge": f"{j}|{inp}", "rule": True,
                            "input": inp, "judge_name": j})
    return out


def select(dev: list[dict], judge_names: list[str], fr_cap: float) -> dict:
    res = []
    for v in variants(judge_names):
        cv = calibrate_constrained(dev, v, fr_cap)
        eligible = v["input"] in (None, *ELIGIBLE_INPUTS)
        res.append({**(cv or v), "admissible": cv is not None, "eligible": eligible,
                    "dev": eval_ext.variant_metrics(dev, cv) if cv else None})
    cands = [i for i, r in enumerate(res) if r["admissible"] and r["eligible"]]
    chosen = max(cands, key=lambda i: (res[i]["dev"]["balanced_accuracy"], -i))
    from .eval import auc
    keys = ["cos"] + [f"{j}|{i}" for j in judge_names for i in INPUTS]
    return {"fr_cap": fr_cap, "variants": res, "chosen": res[chosen]["name"],
            "auc_dev": {k: auc(dev, k) for k in keys}}


# ------------------------------------------------------------------ systems
def system_answerers(lang: str, embedder, judges: dict, chosen: dict) -> dict:
    model = MODELS[lang]
    m3_thr = MODEL_MIN_SCORES[model]
    j4, p4, c4, t4, i4 = M4_ANSWERABILITY_DEFAULTS[model]
    sysm = {
        "M3": (ExtractiveAnswerer(embedder=embedder, min_score=m3_thr, heading_context=False), False),
        "M4": (ExtractiveAnswerer(embedder=embedder, min_score=c4, judge=judges[j4], judge_min=t4, judge_policy=p4,
                                  judge_input=i4), True),
        "M5-fallback (judge none)": (ExtractiveAnswerer(embedder=embedder, min_score=m3_thr), True),
    }
    if chosen.get("judge_name"):
        sysm["M5"] = (ExtractiveAnswerer(embedder=embedder, min_score=chosen["t_cos"] if chosen["t_cos"] is not None
                                         else 0.0, judge=judges[chosen["judge_name"]], judge_min=chosen["t_judge"],
                                         judge_policy=chosen["policy"], judge_input=chosen["input"]), True)
    else:
        sysm["M5"] = (ExtractiveAnswerer(embedder=embedder, min_score=chosen["t_cos"]), True)
    return sysm


def balanced(s: dict) -> float:
    return round(((s["refusal_accuracy"] or 0) + 100 - (s["false_refusal_rate"] or 0)) / 2, 1)


def run_systems(svc, items, sysm: dict, expander) -> dict:
    out = {}
    for name, (ans, use_qe) in sysm.items():
        rows = run_answers(svc, items, ans, TOP_K, expander=expander if use_qe else None)
        s = summarize_answers(rows)
        s["balanced_accuracy"] = balanced(s)
        out[name] = {"summary": s, "rows": rows}
    return out


def fallback_rule(test_sys: dict) -> dict:
    m3, m5 = test_sys["M3"]["summary"], test_sys["M5"]["summary"]
    worse = m5["answer_accuracy"] < m3["answer_accuracy"] or m5["false_refusal_rate"] > m3["false_refusal_rate"]
    return {"triggered": worse, "m3": {k: m3[k] for k in ("answer_accuracy", "false_refusal_rate")},
            "m5": {k: m5[k] for k in ("answer_accuracy", "false_refusal_rate")}}


# ------------------------------------------------------------------ main
def run(lang: str, final: bool, force_retest: bool, model_dir: Path, tag: str = "") -> dict:
    sums = verify_test_frozen()
    model = MODELS[lang]
    out_path = RESULTS / f"m5_{lang}.json" if final else RESULTS / f"m5_{lang}.dev{'.' + tag if tag else ''}.json"
    n_test_runs = 0
    if final and out_path.exists():
        prev = json.loads(out_path.read_text())
        n_test_runs = prev.get("test_evaluations", 0)
        if n_test_runs and not force_retest:
            raise SystemExit(f"{out_path.name}: test was already evaluated {n_test_runs}x; refusing (--force-retest)")
    docs = PROJECT_ROOT / "sample_docs" / lang
    data = {"dev": load_dev(lang)}
    if final:
        data["test"] = load_dataset(DS / f"{lang}.test.jsonl")
    errs = [e for sp in data for e in validate_dataset(data[sp], docs)]
    if errs:
        raise SystemExit("\n".join(errs))
    m4res = json.loads((RESULTS / f"{lang}__{model.replace('/', '_')}.json").read_text())
    alpha = m4res["chunking"][str(CHUNK)]["alpha"]["chosen"]
    qe_mode, qe_model, qe_w = QUERY_EXPANSION_DEFAULTS[model]
    apply_model_cache_env(model_dir)
    with tempfile.TemporaryDirectory() as tmp:
        s = Settings(data_dir=Path(tmp), model_dir=model_dir, embedding_backend="fastembed", embedding_model=model,
                     chunk_size=CHUNK, chunk_overlap=max(CHUNK // 7, 20), top_k=TOP_K, hybrid_alpha=alpha,
                     answerability_judge="none", query_expansion="off", _env_file=None)
        svc = RAGService(s)
        for p in sorted(docs.iterdir()):
            svc.ingest(p.name, p.read_bytes())
        qe = QueryExpander(qe_mode, model=qe_model, weight=qe_w, timeout=60,
                           cache_path=RESULTS / f"qe_cache_{lang}.json")
        judges = {j: build_judge(j, model_dir) for j in JUDGES[lang]}
        ext0 = ExtractiveAnswerer(embedder=svc.embedder, min_score=0.0)
        t0 = time.time()
        feats = {"dev": features(svc, data["dev"], ext0, judges, qe)}
        qe.save_cache()
        print(f"[{lang}] dev features: {len(feats['dev'])} questions in {time.time() - t0:.0f}s")
        # M3 level of dev false refusal = the M3 system on the (extended) dev split
        m3_dev = run_systems(svc, data["dev"], {"M3": system_answerers(lang, svc.embedder, judges,
                                                                       {"t_cos": 0.5})["M3"]}, qe)["M3"]
        fr_cap = m3_dev["summary"]["false_refusal_rate"]
        sel = select(feats["dev"], JUDGES[lang], fr_cap)
        chosen = next(v for v in sel["variants"] if v["name"] == sel["chosen"])
        print(f"[{lang}] dev FR cap (M3) = {fr_cap}; chosen = {sel['chosen']} {chosen['dev']}")
        sysm = system_answerers(lang, svc.embedder, judges, chosen)
        result = {"lang": lang, "model": model, "alpha": alpha, "query_expansion": [qe_mode, qe_model, qe_w],
                  "date": time.strftime("%Y-%m-%d %H:%M %Z"), "test_sha256": sums,
                  "n_questions": {sp: len(v) for sp, v in data.items()},
                  "n_dev_extra": len(load_dataset(DS / f"{lang}.dev_extra.jsonl")),
                  "selection": sel, "chosen_config": {k: chosen.get(k) for k in
                                                      ("name", "judge_name", "policy", "input", "t_cos", "t_judge")},
                  "systems": {"dev": run_systems(svc, data["dev"], sysm, qe)}}
        if final:
            # ---- the single test evaluation: frozen choices, nothing below feeds back into selection
            feats["test"] = features(svc, data["test"], ext0, judges, qe)
            for v in sel["variants"]:
                v["test"] = eval_ext.variant_metrics(feats["test"], v) if v["admissible"] else None
            from .eval import auc
            sel["auc_test"] = {k: auc(feats["test"], k) for k in sel["auc_dev"]}
            result["systems"]["test"] = run_systems(svc, data["test"], sysm, qe)
            result["fallback"] = fallback_rule(result["systems"]["test"]) if lang == "en" else \
                {"applies": False, **fallback_rule(result["systems"]["test"])}
            result["test_evaluations"] = n_test_runs + 1
        qe.save_cache()
        result["features"] = {sp: [{k: v for k, v in r.items() if k != "texts"} for r in rows]
                              for sp, rows in feats.items()}
        result["judge_texts_changed"] = {sp: [{"id": r["id"], **r["texts"]} for r in rows
                                              if len({tuple(t) for t in r["texts"].values()}) > 1]
                                         for sp, rows in feats.items()}
        for sp, sy in result["systems"].items():
            for n, x in sy.items():
                print(f"   {sp} {n}: {x['summary']}")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
    print(f"wrote {out_path}")
    return result


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", choices=["en", "zh"], required=True)
    ap.add_argument("--final", action="store_true", help="evaluate test once with the dev-frozen choices")
    ap.add_argument("--force-retest", action="store_true")
    ap.add_argument("--tag", default="", help="dev-only runs: write m5_{lang}.dev.{tag}.json")
    ap.add_argument("--model-dir", default=os.environ.get("MODEL_DIR", str(PROJECT_ROOT / "models")))
    a = ap.parse_args(argv)
    if a.final and a.tag:
        ap.error("--tag is for dev-only runs")
    run(a.lang, a.final, a.force_retest, Path(a.model_dir), a.tag)


if __name__ == "__main__":
    main()


# ------------------------------------------------------------------ REPORT.md section
def report_lines() -> list[str]:
    files = [RESULTS / f"m5_{lang}.json" for lang in ("en", "zh")]
    res = [json.loads(p.read_text()) for p in files if p.exists()]
    if not res:
        return []
    L = ["## 8. M5: answerability re-selected on an extended dev set under a false-refusal cap", "",
         "Protocol (docs/DESIGN.md section 13, fixed before the single test run): dev = M3/M4 dev + new "
         "leakage-checked unanswerable questions (`eval/datasets/{lang}.dev_extra.jsonl`); judges read the candidate "
         "sentence *with its heading context*; thresholds/variant chosen on dev under *dev false refusal <= M3's*; "
         "test evaluated once (`python -m rag.eval_m5 --final`). Test was already seen in M4, so the M5 test numbers "
         "may carry a slight selection bias. Systems are end-to-end (retrieval with the deployed query expansion, "
         "extractive answer, refusal); latency = CPU wall time incl. the expansion LLM cost.", ""]
    L += ["| lang | system | split | answer acc | false refusal | refusal acc | balanced acc | latency p50 / p95 ms |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res:
        for sp in ("dev", "test"):
            for name in ("M3", "M4", "M5", "M5-fallback (judge none)"):
                s = r["systems"].get(sp, {}).get(name, {}).get("summary")
                if s:
                    L.append(f"| {r['lang']} | {name} | {sp} (n={r['n_questions'][sp]}) | {s['answer_accuracy']} | "
                             f"{s['false_refusal_rate']} | {s['refusal_accuracy']} | {s['balanced_accuracy']} | "
                             f"{s['latency_p50_ms']} / {s['latency_p95_ms']} |")
    L += ["", "Chosen on dev (FR cap = M3's dev false refusal) and the pre-registered English fallback:", ""]
    for r in res:
        c, fb = r["chosen_config"], r.get("fallback", {})
        L.append(f"- **{r['lang']}**: `{c['name']}` (cos {c['t_cos']}, judge {c['t_judge']}), dev FR cap "
                 f"{r['selection']['fr_cap']}; test evaluations: {r.get('test_evaluations')}; fallback rule "
                 + ("not applicable (zh)" if fb.get("applies") is False else
                    f"**{'triggered' if fb.get('triggered') else 'not triggered'}** (M5 test {fb.get('m5')} vs M3 {fb.get('m3')})"))
    L.append("- Shipped defaults (`config.ANSWERABILITY_DEFAULTS`): **en = M5-fallback (judge none, cosine 0.67)** if the "
             "rule triggered, else M5; **zh = M5**. Chunk-order fix after the final run: dev-only re-runs "
             "(`m5_{lang}.dev.postfix.json`) reproduce the final run's dev choices; test was not re-evaluated.")
    L += ["", "Top variants by dev balanced accuracy (admissible = meets the FR cap; plain input = M4 ablation, "
          "not eligible). Test columns are reported only:", "",
          "| lang | variant | thresholds (cos / judge) | dev bal. | dev FR | test bal. | test FR | test refusal acc |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res:
        vs = sorted([v for v in r["selection"]["variants"] if v["admissible"]], key=lambda v: -v["dev"]["balanced_accuracy"])
        for v in vs[:8]:
            t = v.get("test") or {}
            nm = f"**{v['name']}**" if v["name"] == r["selection"]["chosen"] else v["name"] + ("" if v["eligible"] else " (ablation)")
            L.append(f"| {r['lang']} | {nm} | {v.get('t_cos', '–')} / {v.get('t_judge', '–')} | {v['dev']['balanced_accuracy']} "
                     f"| {v['dev']['false_refusal_rate']} | {t.get('balanced_accuracy', '–')} | {t.get('false_refusal_rate', '–')} "
                     f"| {t.get('refusal_accuracy', '–')} |")
    L += ["", "ROC AUC of each raw score by judge input (dev / test):", "",
          "| lang | score | plain | heading | section |", "|---|---|---|---|---|"]
    for r in res:
        a, b = r["selection"]["auc_dev"], r["selection"].get("auc_test", {})
        L.append(f"| {r['lang']} | cos | {a['cos']} / {b.get('cos', '–')} | | |")
        for j in JUDGES[r["lang"]]:
            L.append(f"| {r['lang']} | {j} | " + " | ".join(f"{a[f'{j}|{i}']} / {b.get(f'{j}|{i}', '–')}" for i in INPUTS) + " |")
    return L + [""]
