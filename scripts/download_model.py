"""Download the embedding model into MODEL_DIR (never ~/.cache)."""
from rag.config import get_settings
from rag.embeddings import FastEmbedEmbedder

if __name__ == "__main__":
    s = get_settings()
    print(f"Downloading {s.embedding_model} into {s.model_dir} ...")
    e = FastEmbedEmbedder(s.embedding_model, s.model_dir)
    print(f"OK: {e.name}, dim={e.dim}")
