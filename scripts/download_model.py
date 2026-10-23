"""Download models into MODEL_DIR (never ~/.cache).

    python scripts/download_model.py                      # EMBEDDING_MODEL from settings
    python scripts/download_model.py BAAI/bge-small-zh-v1.5 ...
    python scripts/download_model.py nli reranker-m3      # answerability judges (rag.answerability)
"""
import sys

from rag.answerability import RERANKERS, build_judge
from rag.config import get_settings
from rag.embeddings import FastEmbedEmbedder

if __name__ == "__main__":
    s = get_settings()
    for name in sys.argv[1:] or [s.embedding_model]:
        print(f"Ensuring {name} in {s.model_dir} ...")
        if name in ("nli", "nli-t") or name in RERANKERS:
            j = build_judge(name, s.model_dir)
            print(f"OK: judge {j.name}, sample score {j.score('How long is the warranty?', ['The warranty lasts 3 years.']):.3f}")
            continue
        e = FastEmbedEmbedder(name, s.model_dir)
        print(f"OK: {e.name}, dim={e.dim}")
