"""Download (or export) the models into MODEL_DIR (default ./models; never ~/.cache).

    python scripts/download_model.py --list                # what exists, sizes, which are needed
    python scripts/download_model.py                       # profile "default" (what the shipped defaults use)
    python scripts/download_model.py --profile minimal     # embeddings only (judge off: ANSWERABILITY_JUDGE=none)
    python scripts/download_model.py --profile current     # EMBEDDING_MODEL + the judge it resolves to (Docker bake)
    python scripts/download_model.py --profile full        # + every judge compared in the eval (make eval-m4 / eval-m5)
    python scripts/download_model.py --ollama 0.5b,1.5b    # + local LLMs via scripts/ollama.sh (optional)
    python scripts/download_model.py BAAI/bge-small-zh-v1.5 nli-t ...   # explicit names
    python scripts/download_model.py --profile default --export build/models.tar   # offline bundle

Models are pulled from Hugging Face on first use anyway; this script just does it up front (and
checks each one loads). Sizes are the download sizes measured on this project's machine.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rag.answerability import RERANKERS, build_judge  # noqa: E402
from rag.config import ANSWERABILITY_DEFAULTS, QUERY_EXPANSION_DEFAULTS, get_settings  # noqa: E402

# name -> (approx. download MB, HF repo dir under MODEL_DIR, purpose)
CATALOG: dict[str, tuple[int, str, str]] = {
    "BAAI/bge-small-en-v1.5": (67, "models--Qdrant--bge-small-en-v1.5-onnx-Q", "English embeddings (required for en)"),
    "BAAI/bge-small-zh-v1.5": (96, "models--Qdrant--bge-small-zh-v1.5", "Chinese embeddings (required for zh)"),
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        (235, "models--Qdrant--paraphrase-multilingual-MiniLM-L12-v2-onnx-Q", "multilingual embeddings (eval comparison)"),
    "nli": (1135, "models--Xenova--mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "answerability judge: mDeBERTa XNLI fp32"),
    "reranker": (135, "models--Xenova--ms-marco-MiniLM-L-12-v2", "answerability judge: ms-marco MiniLM (en)"),
    "reranker-m3": (590, "models--onnx-community--bge-reranker-v2-m3-ONNX", "answerability judge: bge-reranker-v2-m3 uint8"),
}
ALIASES = {"nli-t": "nli"}  # same model, different hypothesis template
OLLAMA = {"0.5b": 491, "1.5b": 1100}  # qwen2.5 Q4_K_M via `ollama pull`, MB


def default_set() -> list[str]:
    """Exactly what the shipped defaults need: both embedding models + the judges they resolve to."""
    names = ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5"]
    for model, cfg in ANSWERABILITY_DEFAULTS.items():
        if cfg[0] != "none" and model in names:
            names.append(cfg[0])
    return list(dict.fromkeys(names))


def current_set() -> list[str]:
    """Only what the configured EMBEDDING_MODEL (+ its resolved judge) needs, e.g. for a baked image."""
    s = get_settings()
    return [s.embedding_model] + ([s.resolved_judge] if s.resolved_judge != "none" else [])


PROFILES = {
    "current": current_set,
    "minimal": lambda: ["BAAI/bge-small-en-v1.5", "BAAI/bge-small-zh-v1.5"],
    "default": default_set,
    "full": lambda: [*default_set(), "nli", "nli-t", "reranker", "reranker-m3",
                     "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
}


def key(name: str) -> str:
    return ALIASES.get(name, name)


def ensure(name: str, model_dir: Path) -> None:
    print(f"Ensuring {name} in {model_dir} ...", flush=True)
    if name in ("nli", "nli-t") or name in RERANKERS:
        j = build_judge(name, model_dir)
        print(f"OK: judge {j.name}, sample score {j.score('How long is the warranty?', ['The warranty lasts 3 years.']):.3f}")
        return
    from rag.embeddings import FastEmbedEmbedder
    e = FastEmbedEmbedder(name, model_dir)
    print(f"OK: {e.name}, dim={e.dim}")


def listing(model_dir: Path) -> None:
    need = {key(n) for n in default_set()}
    print(f"MODEL_DIR = {model_dir}\n")
    print(f"{'model':62} {'MB':>6}  {'default':7} {'present':7}  purpose")
    for name, (mb, d, purpose) in CATALOG.items():
        print(f"{name:62} {mb:>6}  {'yes' if name in need else '-':7} {'yes' if (model_dir / d).exists() else '-':7}  {purpose}")
    for tag, mb in OLLAMA.items():
        present = any((model_dir / "ollama" / m / "library" / "qwen2.5" / tag).exists()
                      for m in ("manifests/registry.ollama.ai", "manifests-v2/ollama.com"))
        print(f"{'ollama qwen2.5:' + tag:62} {mb:>6}  {'-':7} {'yes' if present else '-':7}  optional local LLM "
              f"(LLM_BACKEND=openai, QUERY_EXPANSION)")
    tot = sum(CATALOG[key(n)][0] for n in dict.fromkeys(key(n) for n in default_set()))
    print(f"\ndefault profile: ~{tot} MB; minimal: ~{CATALOG['BAAI/bge-small-en-v1.5'][0] + CATALOG['BAAI/bge-small-zh-v1.5'][0]} MB")
    qe = sorted({v[1] for v in QUERY_EXPANSION_DEFAULTS.values()})
    print(f"query expansion defaults use {', '.join(qe)} if an Ollama server is running (otherwise it is skipped).")


def export(names: list[str], model_dir: Path, out: Path) -> None:
    """Tar the HF cache dirs of `names` (symlinks to blobs dereferenced) for an offline machine:
    `tar -xf models.tar -C models/` there."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w", dereference=True) as tar:
        for n in dict.fromkeys(key(x) for x in names):
            d = model_dir / CATALOG[n][1] / "snapshots"
            for f in sorted(p for p in d.rglob("*") if p.is_file()):
                tar.add(f, arcname=str(f.relative_to(model_dir)))
            for ref in (model_dir / CATALOG[n][1] / "refs").glob("*"):
                tar.add(ref, arcname=str(ref.relative_to(model_dir)))
    print(f"wrote {out} ({out.stat().st_size / 1e6:.0f} MB)")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*")
    ap.add_argument("--profile", choices=sorted(PROFILES))
    ap.add_argument("--ollama", default="", help="comma-separated qwen2.5 sizes to pull, e.g. 0.5b,1.5b")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--export", type=Path, help="write a tar of the selected models (after downloading)")
    a = ap.parse_args(argv)
    s = get_settings()
    if a.list:
        listing(s.model_dir)
        return
    names = a.names or PROFILES[a.profile or "default"]()
    if a.profile and a.names:
        names = [*PROFILES[a.profile](), *a.names]
    for n in dict.fromkeys(names):
        ensure(n, s.model_dir)
    for tag in [t for t in a.ollama.split(",") if t]:
        subprocess.run([str(ROOT / "scripts" / "ollama.sh"), "pull", tag], check=True)
    if a.export:
        export(names, s.model_dir, a.export)


if __name__ == "__main__":
    main(sys.argv[1:])
