"""Settings loaded from environment / .env.

Model caches are forced into MODEL_DIR so nothing lands in ~/.cache.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    model_dir: Path = PROJECT_ROOT / "models"
    data_dir: Path = PROJECT_ROOT / "data"

    # embedding: "fastembed" (ONNX, CPU) or "hash" (deterministic, offline, for tests)
    embedding_backend: str = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"

    chunk_size: int = 800        # characters
    chunk_overlap: int = 120     # characters
    top_k: int = 4
    hybrid_alpha: float = 0.5    # weight of vector vs BM25 in RRF fusion (0..1)
    max_upload_mb: int = 20
    # extractive answerer: minimum sentence score (0.5*lexical + cosine) to answer at all.
    # 0.58 is calibrated for bge-small; unrelated questions score ~0.5.
    answer_min_score: float = 0.58

    # answerer: "extractive" (default, no model) or "openai" (any OpenAI-compatible
    # endpoint: Ollama, llama.cpp server, vLLM, LM Studio ...)
    llm_backend: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:0.5b"
    llm_api_key: str = "not-needed"
    llm_timeout: float = 60.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "rag.sqlite3"


def apply_model_cache_env(model_dir: Path) -> None:
    """Point every HF-style cache variable at MODEL_DIR (never ~/.cache)."""
    model_dir.mkdir(parents=True, exist_ok=True)
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE",
                "SENTENCE_TRANSFORMERS_HOME", "FASTEMBED_CACHE_PATH", "TRANSFORMERS_CACHE"):
        os.environ[var] = str(model_dir)


def get_settings() -> Settings:
    s = Settings()
    apply_model_cache_env(s.model_dir)
    s.data_dir.mkdir(parents=True, exist_ok=True)
    return s
