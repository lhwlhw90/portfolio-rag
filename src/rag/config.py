"""Settings loaded from environment / .env.

Model caches are forced into MODEL_DIR so nothing lands in ~/.cache.
"""
from __future__ import annotations

import os
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# Refusal thresholds calibrated on the eval *dev* split only (chunk 800; see
# eval/results/REPORT.md, section 3). Cosine scales differ per model, so the
# threshold is model-specific. Test-split numbers are reported, never tuned on.
DEFAULT_MIN_SCORE = 0.65
MODEL_MIN_SCORES: dict[str, float] = {
    "BAAI/bge-small-en-v1.5": 0.67,
    "BAAI/bge-small-zh-v1.5": 0.63,
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2": 0.57,  # dev: en 0.59 / zh 0.555 -> midpoint
}


# M4 defaults chosen on the eval *dev* split by `python -m rag.eval --judges ... --qe ...`
# (REPORT.md sections 5 and 6). ANSWERABILITY_JUDGE=auto / QUERY_EXPANSION=auto resolve to these
# per embedding model; models not listed get the M3 behaviour (cosine threshold, no expansion).
# answerability: embedding model -> (judge, policy, cosine threshold, judge threshold)
ANSWERABILITY_DEFAULTS: dict[str, tuple[str, str, float, float]] = {
    "BAAI/bge-small-en-v1.5": ("nli-t", "and", 0.665, 0.6058),       # dev bal. acc 77.9 -> 81.1
    "BAAI/bge-small-zh-v1.5": ("reranker-m3", "and", 0.575, 0.1182),  # dev bal. acc 79.4 -> 86.3
}
# query expansion: embedding model -> (mode, LLM, expansion weight)
QUERY_EXPANSION_DEFAULTS: dict[str, tuple[str, str, float]] = {
    "BAAI/bge-small-en-v1.5": ("rewrite", "qwen2.5:1.5b", 1.0),  # dev MRR 0.892 -> 0.917
    "BAAI/bge-small-zh-v1.5": ("rewrite", "qwen2.5:0.5b", 1.0),  # dev MRR 0.943 -> 0.951
}
DEFAULT_JUDGE_MIN = 0.5


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
    # extractive answerer: minimum best-sentence score (0.5*lexical + cosine) to answer at all.
    # None -> per-model default calibrated by `make eval` (see MODEL_MIN_SCORES).
    answer_min_score: float | None = None

    # answerer: "extractive" (default, no model) or "openai" (any OpenAI-compatible
    # endpoint: Ollama, llama.cpp server, vLLM, LM Studio ...)
    llm_backend: str = "extractive"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_model: str = "qwen2.5:0.5b"
    llm_api_key: str = "not-needed"
    llm_timeout: float = 60.0
    llm_gate: bool = True  # refuse before calling the LLM when extractive confidence < threshold

    # extractive answers: prefix the section heading when it carries the asked-for date / version
    heading_context: bool = True
    # answerability judge on top of / instead of the cosine threshold:
    # auto (dev-chosen per embedding model, ANSWERABILITY_DEFAULTS) | none | nli | reranker | reranker-m3
    answerability_judge: str = "auto"
    answerability_policy: str | None = None    # replace | and | or  (see ExtractiveAnswerer); None -> auto/"and"
    answerability_min: float | None = None     # judge threshold; None -> auto / DEFAULT_JUDGE_MIN
    # query expansion for paraphrased questions: auto | off | prf | rewrite | hyde (rewrite/hyde call
    # the OpenAI-compatible LLM at LLM_BASE_URL, e.g. Ollama; failures fall back to no expansion)
    query_expansion: str = "auto"
    query_expansion_model: str | None = None   # None -> auto / qwen2.5:1.5b
    query_expansion_weight: float | None = None
    query_expansion_timeout: float = 20.0

    def _auto_judge(self) -> tuple[str, str, float, float] | None:
        if self.answerability_judge != "auto" or self.embedding_backend == "hash":
            return None
        return ANSWERABILITY_DEFAULTS.get(self.embedding_model)

    @property
    def resolved_judge(self) -> str:
        auto = self._auto_judge()
        return auto[0] if auto else ("none" if self.answerability_judge == "auto" else self.answerability_judge)

    @property
    def resolved_policy(self) -> str:
        auto = self._auto_judge()
        return self.answerability_policy or (auto[1] if auto else "and")

    @property
    def resolved_judge_min(self) -> float:
        auto = self._auto_judge()
        if self.answerability_min is not None:
            return self.answerability_min
        return auto[3] if auto else DEFAULT_JUDGE_MIN

    @property
    def resolved_min_score(self) -> float:
        if self.answer_min_score is not None:
            return self.answer_min_score
        if self.embedding_backend == "hash":
            return 0.0
        auto = self._auto_judge()  # the cosine threshold is calibrated jointly with the judge
        return auto[2] if auto else MODEL_MIN_SCORES.get(self.embedding_model, DEFAULT_MIN_SCORE)

    @property
    def resolved_expansion(self) -> tuple[str, str, float]:
        """(mode, model, weight) after resolving QUERY_EXPANSION=auto."""
        auto = (QUERY_EXPANSION_DEFAULTS.get(self.embedding_model)
                if self.query_expansion == "auto" and self.embedding_backend != "hash" else None)
        mode = auto[0] if auto else ("off" if self.query_expansion == "auto" else self.query_expansion)
        model = self.query_expansion_model or (auto[1] if auto else "qwen2.5:1.5b")
        weight = self.query_expansion_weight if self.query_expansion_weight is not None else (auto[2] if auto else 1.0)
        return mode, model, weight

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
