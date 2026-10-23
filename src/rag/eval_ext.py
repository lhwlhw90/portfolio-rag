"""M4 evaluation blocks used by rag.eval: answerability judges, query expansion and heading
context. Same protocol as M3: everything is chosen on dev, test is reported only."""
from __future__ import annotations

import statistics
import time

from .answerability import TimedJudge
from .answerer import ExtractiveAnswerer
from .query import QueryExpander

RULE_REASONS = ("no_context", "no_candidates", "unknown_identifier")


def _p50(xs):
    return round(statistics.median(xs), 1) if xs else None


def _p95(xs):
    xs = sorted(xs)
    return round(xs[min(len(xs) - 1, int(0.95 * len(xs)))], 1) if xs else None


def pct(x: float) -> float:
    return round(100 * x, 1)


# ---------------------------------------------------------------- answerability
def answerability_features(svc, items: list[dict], judges: dict, top_k: int, embedder, answer_correct) -> list[dict]:
    """Per question: cosine confidence, rule refusals, judge scores (+ judge latency) and whether
    the (threshold-free) extractive answer contains the expected fact."""
    ext0 = ExtractiveAnswerer(embedder=embedder, min_score=0.0)
    timed = {n: TimedJudge(j) for n, j in judges.items()}
    rows = []
    for it in items:
        t0 = time.perf_counter()
        hits = svc.retriever.retrieve(it["question"], k=top_k, mode="hybrid")
        ans = ext0.answer(it["question"], hits)
        base_ms = (time.perf_counter() - t0) * 1000
        cands = ext0.score_sentences(it["question"], hits)[:3]
        f = {"id": it["id"], "type": it["type"], "cos": ans.confidence or 0.0, "base_ms": round(base_ms, 1),
             "rule": ans.refusal_reason if ans.refused and ans.refusal_reason in RULE_REASONS else None,
             "correct": bool(it["answer_contains"]) and answer_correct(ans.answer, it["answer_contains"])}
        for n, j in timed.items():
            f[n] = round(j.score(it["question"], [c[3] for c in cands]), 4) if cands else 0.0
            f[f"{n}_ms"] = round(j.times_ms[-1], 1) if cands else 0.0
        rows.append(f)
    return rows


def _decide(r: dict, v: dict) -> bool:
    """True = refuse under variant v."""
    if v["rule"] and r["rule"]:
        return True
    if not v["rule"] and r["rule"] in ("no_context", "no_candidates"):
        return True
    cos_ok = r["cos"] >= v["t_cos"] if v.get("t_cos") is not None else None
    j_ok = r[v["judge"]] >= v["t_judge"] if v.get("judge") else None
    if v["policy"] == "cosine":
        return not cos_ok
    if v["policy"] == "replace":
        return not j_ok
    if v["policy"] == "and":
        return not (cos_ok and j_ok)
    return not (cos_ok or j_ok)  # "or"


def variant_metrics(rows: list[dict], v: dict) -> dict:
    ans = [r for r in rows if r["type"] != "unanswerable"]
    una = [r for r in rows if r["type"] == "unanswerable"]
    ra = sum(_decide(r, v) for r in una) / len(una)
    fr = sum(_decide(r, v) for r in ans) / len(ans)
    acc = sum(r["correct"] and not _decide(r, v) for r in ans) / len(ans)
    judge_ms = [r[f"{v['judge']}_ms"] for r in rows] if v.get("judge") else [0.0] * len(rows)
    total = [r["base_ms"] + m for r, m in zip(rows, judge_ms)]
    return {"refusal_accuracy": pct(ra), "false_refusal_rate": pct(fr), "balanced_accuracy": pct((ra + 1 - fr) / 2),
            "answer_accuracy": pct(acc), "judge_ms_p50": _p50(judge_ms), "latency_p50_ms": _p50(total),
            "latency_p95_ms": _p95(total)}


def _grid(values: list[float], lo: float | None = None, hi: float | None = None) -> list[float]:
    """Candidate thresholds: midpoints between consecutive distinct dev values (+ the ends)."""
    xs = sorted(set(round(x, 4) for x in values))
    mids = [round((a + b) / 2, 4) for a, b in zip(xs, xs[1:])]
    out = [xs[0] - 1e-3] + mids + [xs[-1] + 1e-3] if xs else [0.5]
    return [x for x in out if (lo is None or x >= lo) and (hi is None or x <= hi)]


def calibrate_variant(dev: list[dict], v: dict) -> dict:
    """Pick threshold(s) maximizing dev balanced accuracy; ties -> middle of the plateau."""
    cos_grid = [t / 200 for t in range(80, 181)] if v["policy"] != "replace" else [None]
    j_grid = _grid([r[v["judge"]] for r in dev]) if v.get("judge") else [None]
    scored = []
    for tc in cos_grid:
        for tj in j_grid:
            m = variant_metrics(dev, {**v, "t_cos": tc, "t_judge": tj})
            scored.append((m["balanced_accuracy"], tc, tj))
    best = max(s[0] for s in scored)
    plateau = [s for s in scored if s[0] == best]
    _, tc, tj = plateau[len(plateau) // 2]
    return {**v, "t_cos": tc, "t_judge": tj}


def eval_answerability(feats: dict, judge_names: list[str]) -> dict:
    """Variants calibrated on feats['dev'], reported on dev and test."""
    variants = [{"name": "M3: cosine + identifier rule", "policy": "cosine", "rule": True},
                {"name": "cosine only (no identifier rule)", "policy": "cosine", "rule": False}]
    for j in judge_names:
        variants += [{"name": f"{j} only + identifier rule", "policy": "replace", "judge": j, "rule": True},
                     {"name": f"cosine AND {j} + identifier rule", "policy": "and", "judge": j, "rule": True},
                     {"name": f"cosine OR {j} + identifier rule", "policy": "or", "judge": j, "rule": True}]
    out = []
    for v in variants:
        cv = calibrate_variant(feats["dev"], v)
        out.append({**cv, "dev": variant_metrics(feats["dev"], cv), "test": variant_metrics(feats["test"], cv)})
    # default = best dev balanced accuracy; ties -> the simplest (earliest) variant
    chosen = max(range(len(out)), key=lambda i: (out[i]["dev"]["balanced_accuracy"], -i))
    from .eval import auc  # noqa: avoid a cycle at import time
    aucs = {sp: {k: auc(feats[sp], k) for k in ["cos", *judge_names]} for sp in feats}
    return {"variants": out, "chosen": out[chosen]["name"], "auc": aucs}


# ---------------------------------------------------------------- query expansion
def eval_query_expansion(svc, data: dict, configs: list[dict], eval_retrieval, weights=(0.5, 1.0),
                         cache_path=None) -> dict:
    """configs: [{"mode": "prf"}, {"mode": "rewrite", "model": "qwen2.5:1.5b"}, ...]. The weight of
    the expansions is tuned on dev per config; the default config is the best dev MRR."""
    res = [{"name": "off", "mode": "off", "weight": None,
            **{sp: eval_retrieval(svc, data[sp], "hybrid") for sp in data}}]
    for c in configs:
        qe = QueryExpander(c["mode"], base_url=c.get("url", "http://127.0.0.1:11434/v1"),
                           model=c.get("model") or "", timeout=60, cache_path=cache_path)
        name = c["mode"] + (f" ({c['model']})" if c.get("model") else "")
        first: dict = {}
        for sp in data:  # first pass at weight 1.0 also measures the (uncached) expansion latency
            qe.weight = 1.0
            first[sp] = eval_retrieval(svc, data[sp], "hybrid", expander=qe)
        lat = {sp: [r["expansion_ms"] for r in first[sp]["rows"]] for sp in data}
        best_w, best = None, None
        for w in weights:
            qe.weight = w
            r = first["dev"] if w == 1.0 else eval_retrieval(svc, data["dev"], "hybrid", expander=qe)
            if best is None or r["overall"]["mrr"] > best["overall"]["mrr"]:
                best_w, best = w, r
        qe.weight = best_w
        entry = {"name": name, "mode": c["mode"], "model": c.get("model"), "weight": best_w}
        for sp in data:
            entry[sp] = first[sp] if best_w == 1.0 else eval_retrieval(svc, data[sp], "hybrid", expander=qe)
            entry[sp]["expansion_ms_p50"] = _p50(lat[sp])
            entry[sp]["expansion_ms_p95"] = _p95(lat[sp])
        res.append(entry)
        qe.save_cache()
    chosen = max(range(len(res)), key=lambda i: (res[i]["dev"]["overall"]["mrr"], -i))
    return {"configs": res, "chosen": res[chosen]["name"]}


# ---------------------------------------------------------------- heading context
def eval_heading(svc, data: dict, embedder, thr: float, top_k: int, run_answers, summarize) -> dict:
    out = {}
    for flag in (False, True):
        ext = ExtractiveAnswerer(embedder=embedder, min_score=thr, heading_context=flag)
        out["on" if flag else "off"] = {sp: run_answers(svc, data[sp], ext, top_k) for sp in data}
    res = {}
    for sp in data:
        off, on = out["off"][sp], out["on"][sp]
        changed = [{"id": a["id"], "question": a["question"], "off": a["answer"], "on": b["answer"],
                    "correct_off": a["answer_correct"], "correct_on": b["answer_correct"]}
                   for a, b in zip(off, on) if a["answer"] != b["answer"]]
        res[sp] = {"off": summarize(off)["answer_accuracy"], "on": summarize(on)["answer_accuracy"],
                   "n_changed": len(changed), "changed": changed}
    return res


# ---------------------------------------------------------------- end-to-end M4 system
def eval_system(svc, data: dict, embedder, entry: dict, judges: dict, top_k: int, run_answers, summarize,
                cache_path=None) -> dict:
    """Real answerer runs with the dev-chosen answerability variant + query expansion + headings."""
    a = next(v for v in entry["answerability"]["variants"] if v["name"] == entry["answerability"]["chosen"])
    q = next(c for c in entry["query_expansion"]["configs"] if c["name"] == entry["query_expansion"]["chosen"])
    judge = judges.get(a.get("judge")) if a.get("judge") else None
    ext = ExtractiveAnswerer(embedder=embedder, min_score=a["t_cos"] if a.get("t_cos") is not None else 0.0,
                             check_identifiers=a["rule"], heading_context=True, judge=judge,
                             judge_min=a["t_judge"] if a.get("t_judge") is not None else 0.5,
                             judge_policy=a["policy"] if a["policy"] != "cosine" else "and", judge_input="plain")
    qe = (QueryExpander(q["mode"], model=q.get("model") or "", weight=q["weight"] or 1.0, cache_path=cache_path,
                        timeout=60) if q["mode"] != "off" else None)
    out = {"answerability": a["name"], "query_expansion": q["name"]}
    for sp in data:
        rows = run_answers(svc, data[sp], ext, top_k, expander=qe)
        out[sp] = {"summary": summarize(rows), "rows": rows}
    if qe is not None:
        qe.save_cache()
    return out
