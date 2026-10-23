"""Build the M5 dev-only extension eval/datasets/{lang}.dev_extra.jsonl from
eval/authoring/{lang}_dev_extra.py, with leakage checks:

1. the frozen test files must match eval/datasets/SHA256SUMS (= the M3/M4 test sets, unchanged);
2. no new question may duplicate / closely paraphrase a test question or an existing dev question
   (rag.leakage: char-3-gram cosine < 0.6 and content-token Jaccard < 0.5);
3. optional (--embed): report the max bge-small embedding cosine to any test question and fail
   above --embed-max (default 0.92).

    python scripts/build_dev_extra.py en zh [--embed] [--force]
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from rag.leakage import check, nearest, sha256  # noqa: E402

DS = ROOT / "eval" / "datasets"
EMBED = {"en": "BAAI/bge-small-en-v1.5", "zh": "BAAI/bge-small-zh-v1.5"}


def frozen_sums() -> dict[str, str]:
    return dict(reversed(line.split()) for line in (DS / "SHA256SUMS").read_text().splitlines() if line.strip())


def verify_test_frozen() -> None:
    sums = frozen_sums()
    for name, digest in sums.items():
        actual = sha256(DS / name)
        if actual != digest:
            raise SystemExit(f"{name} changed: {actual} != frozen {digest}")


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def build(lang: str) -> list[dict]:
    path = ROOT / "eval" / "authoring" / f"{lang}_dev_extra.py"
    spec = importlib.util.spec_from_file_location(f"{lang}_dev_extra", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    counters: dict[str, int] = {}
    rows = []
    for qtype, question, evidence, contains in mod.Q:
        counters[qtype] = counters.get(qtype, 0) + 1
        rows.append({"id": f"{lang}-x{qtype[0]}{counters[qtype]:03d}", "type": qtype, "question": question,
                     "evidence": [{"doc": d, "text": t} for d, t in evidence], "answer_contains": contains,
                     "source": "M5 dev extension"})
    return rows


def main(argv: list[str]) -> None:
    langs = [a for a in argv if not a.startswith("--")] or ["en", "zh"]
    verify_test_frozen()
    print("test files unchanged (SHA256SUMS ok)")
    for lang in langs:
        rows = build(lang)
        test, dev = load_rows(DS / f"{lang}.test.jsonl"), load_rows(DS / f"{lang}.dev.jsonl")
        errs = check(rows, test + dev)
        if errs:
            raise SystemExit(f"[{lang}] leakage check failed:\n" + "\n".join(errs))
        near = [nearest(r["question"], test) for r in rows]
        mc = max(near, key=lambda b: b["char"][0])["char"]
        mj = max(near, key=lambda b: b["jaccard"][0])["jaccard"]
        print(f"[{lang}] {len(rows)} new dev questions; max vs test: char-3gram cosine {mc[0]} ({mc[2]!r}), "
              f"token Jaccard {mj[0]} ({mj[2]!r})")
        if "--embed" in argv:
            from rag.config import PROJECT_ROOT, apply_model_cache_env
            from rag.embeddings import build_embedder
            apply_model_cache_env(PROJECT_ROOT / "models")
            emb = build_embedder("fastembed", EMBED[lang], PROJECT_ROOT / "models")
            sims = emb.embed_documents([r["question"] for r in rows]) @ emb.embed_documents(
                [t["question"] for t in test]).T
            i, j = divmod(int(sims.argmax()), sims.shape[1])
            limit = float(next((a.split("=")[1] for a in argv if a.startswith("--embed-max=")), 0.92))
            print(f"[{lang}] max embedding cosine vs test {sims.max():.3f}: {rows[i]['question']!r} ~ "
                  f"{test[j]['question']!r} (limit {limit})")
            if sims.max() >= limit:
                raise SystemExit(f"[{lang}] embedding near-duplicate above {limit}")
        out = DS / f"{lang}.dev_extra.jsonl"
        if out.exists() and "--force" not in argv:
            old = out.read_text(encoding="utf-8")
            new = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
            if old != new:
                raise SystemExit(f"{out.name} exists and differs; re-run with --force")
        out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(sys.argv[1:])
