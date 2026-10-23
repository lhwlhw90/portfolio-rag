"""Download embedding models into MODEL_DIR (never ~/.cache).

    python scripts/download_model.py                      # EMBEDDING_MODEL from settings
    python scripts/download_model.py BAAI/bge-small-zh-v1.5 ...
"""
import sys

from rag.config import get_settings
from rag.embeddings import FastEmbedEmbedder

if __name__ == "__main__":
    s = get_settings()
    for name in sys.argv[1:] or [s.embedding_model]:
        print(f"Ensuring {name} in {s.model_dir} ...")
        e = FastEmbedEmbedder(name, s.model_dir)
        print(f"OK: {e.name}, dim={e.dim}")
