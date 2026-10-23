"""Build the frozen dev/test eval splits from eval/authoring/{lang}_questions.py.

The split is stratified by question type and deterministic (sha1 of the question id), with
DEV_FRACTION of every type going to dev. Thresholds / hybrid weights are tuned on dev only;
test is the held-out set used for the headline numbers.

The output files are committed. Re-running refuses to overwrite them unless --force is given,
so the test set cannot drift silently.

    python scripts/split_dataset.py en zh [--force]
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEV_FRACTION = 0.4
TYPES = ("factual", "paraphrase", "cross", "unanswerable")


def load(lang: str):
    path = ROOT / "eval" / "authoring" / f"{lang}_questions.py"
    spec = importlib.util.spec_from_file_location(f"{lang}_questions", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows, counters = [], {t: 0 for t in TYPES}
    for qtype, question, evidence, contains in mod.Q:
        counters[qtype] += 1
        rows.append({
            "id": f"{lang}-{qtype[0]}{counters[qtype]:03d}",
            "type": qtype,
            "question": question,
            "evidence": [{"doc": mod.D[k], "text": t} for k, t in evidence],
            "answer_contains": contains,
        })
    return rows


def split(rows):
    dev, test = [], []
    for t in TYPES:
        group = sorted((r for r in rows if r["type"] == t), key=lambda r: hashlib.sha1(r["id"].encode()).hexdigest())
        n_dev = round(len(group) * DEV_FRACTION)
        dev += group[:n_dev]
        test += group[n_dev:]
    key = lambda r: (TYPES.index(r["type"]), r["id"])
    return sorted(dev, key=key), sorted(test, key=key)


def main(argv):
    force = "--force" in argv
    for lang in [a for a in argv if not a.startswith("--")] or ["en", "zh"]:
        dev, test = split(load(lang))
        for name, part in (("dev", dev), ("test", test)):
            out = ROOT / "eval" / "datasets" / f"{lang}.{name}.jsonl"
            if out.exists() and not force:
                print(f"{out.relative_to(ROOT)} exists (frozen); use --force to regenerate")
                continue
            out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in part), encoding="utf-8")
            print(f"wrote {out.relative_to(ROOT)}: {len(part)} questions")


if __name__ == "__main__":
    main(sys.argv[1:])
