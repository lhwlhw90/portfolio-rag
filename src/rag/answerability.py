"""Answerability judges: does the retrieved evidence actually state what the question asks?

The M3 refusal signal is max cosine(sentence, question). Cosine measures topical similarity,
so "What is the price of the X2?" scores high against any X2 sentence. M4 adds optional
second opinions that read question and evidence *together* (cross-attention):

- ``NLIJudge``: multilingual mDeBERTa-v3 XNLI (fp32 ONNX, 1.1 GB, CPU). Premise = candidate
  sentence, hypothesis = the question ("nli") or "The text answers the question: {q}" ("nli-t");
  score = max P(entailment) over the top candidate sentences.
- ``CrossEncoderJudge``: a passage reranker (fastembed ONNX: ms-marco-MiniLM-L-12 fp32 for English,
  bge-reranker-v2-m3 uint8 multilingual) used as a relevance classifier; score = sigmoid(max logit).

Judges only produce a score; the policy (replace cosine, require both, accept either) and the
thresholds live in ExtractiveAnswerer and are calibrated on the eval *dev* split only
(``python -m rag.eval``; see eval/results/REPORT.md section 5). All models live in MODEL_DIR.
"""
from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Protocol

import numpy as np

from .text import is_cjk

NLI_MODEL = "Xenova/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
# fp32, not int8: (1) on CPUs with AMX, onnxruntime's int8 (U8S8) GEMM uses AMX tiles, and on VMs
# that don't preserve tile data across context switches (onnxruntime issue #33247 - this box is one)
# int8 models return silently *random* results (same pair: P(entail) 0.13..0.62 across calls);
# (2) the 8-bit exports of this model lose most of its accuracy (uint8 vs fp32 on probe pairs:
# r = 0.47; 'part-time' near miss 0.30 vs 0.003). fp32 costs ~90 ms per 3 sentences on CPU.
# uint8 weights (U8U8 -> AVX-512 VNNI) are deterministic, so the multilingual reranker uses uint8
# (it matches its fp32 export within 0.05 on the probe pairs). See docs/DESIGN.md section 12.
NLI_FILE = "onnx/model.onnx"
M3_UINT8 = "BAAI/bge-reranker-v2-m3-uint8"
RERANKERS = {"reranker": "Xenova/ms-marco-MiniLM-L-12-v2", "reranker-m3": M3_UINT8}
JUDGES = ("none", "nli", "nli-t", *RERANKERS)
POLICIES = ("replace", "and", "or")
# hypothesis = the question itself ("nli") or a declarative template ("nli-t"); compared on dev
TEMPLATES = {"en": "The text answers the question: {q}", "zh": "这段文字回答了这个问题：{q}"}


class Judge(Protocol):
    name: str

    def score(self, question: str, sentences: list[str]) -> float: ...


def _snapshot(model_dir: Path, repo: str, filename: str) -> Path:
    """Local snapshot dir of an HF repo inside MODEL_DIR (downloads on first use)."""
    from huggingface_hub import hf_hub_download

    files = [filename, "tokenizer.json", "config.json"]
    paths = [Path(hf_hub_download(repo, f, cache_dir=str(model_dir))) for f in files]
    return paths[1].parent


class NLIJudge:
    def __init__(self, model_dir: Path, repo: str = NLI_MODEL, filename: str = NLI_FILE, threads: int = 4,
                 max_len: int = 256, template: bool = False):
        self.template = template
        self.name = "nli-t" if template else "nli"
        import onnxruntime as ort
        from tokenizers import Tokenizer

        snap = _snapshot(Path(model_dir), repo, filename)
        self.tok = Tokenizer.from_file(str(snap / "tokenizer.json"))
        self.tok.enable_truncation(max_len)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(snap / filename), so, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.sess.get_inputs()}
        import json
        labels = json.loads((snap / "config.json").read_text())["id2label"]
        self.entail = next(int(k) for k, v in labels.items() if v.lower().startswith("entail"))

    def probs(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        enc = self.tok.encode_batch(pairs)
        n = max(len(e.ids) for e in enc)
        pad = lambda xs: np.array([x + [0] * (n - len(x)) for x in xs], dtype=np.int64)
        feed = {"input_ids": pad([e.ids for e in enc]), "attention_mask": pad([e.attention_mask for e in enc])}
        if "token_type_ids" in self.inputs:
            feed["token_type_ids"] = pad([e.type_ids for e in enc])
        logits = self.sess.run(None, feed)[0]
        p = np.exp(logits - logits.max(1, keepdims=True))
        return p / p.sum(1, keepdims=True)

    def hypothesis(self, question: str) -> str:
        if not self.template:
            return question
        return TEMPLATES["zh" if is_cjk(question) else "en"].format(q=question)

    def score(self, question: str, sentences: list[str]) -> float:
        if not sentences:
            return 0.0
        h = self.hypothesis(question)
        # one pair per call, so a sentence's score never depends on how its neighbours are padded
        return float(max(self.probs([(s, h)])[0, self.entail] for s in sentences))


def _register_m3_uint8() -> None:
    from fastembed.common.model_description import ModelSource
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    if not any(m["model"] == M3_UINT8 for m in TextCrossEncoder.list_supported_models()):
        TextCrossEncoder.add_custom_model(M3_UINT8, sources=ModelSource(hf="onnx-community/bge-reranker-v2-m3-ONNX"),
                                          model_file="onnx/model_uint8.onnx", size_in_gb=0.57,
                                          description="bge-reranker-v2-m3, uint8 ONNX")


class CrossEncoderJudge:
    def __init__(self, model_dir: Path, model: str, name: str = "reranker", threads: int = 4):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        if model == M3_UINT8:
            _register_m3_uint8()
        self.name = name
        self.enc = TextCrossEncoder(model, cache_dir=str(model_dir), threads=threads)

    def score(self, question: str, sentences: list[str]) -> float:
        if not sentences:
            return 0.0
        logit = max(float(x) for x in self.enc.rerank(question, sentences, batch_size=1))  # no padding effects
        return 1 / (1 + math.exp(-logit))


class TimedJudge:
    """Wraps a judge and records the wall time of each call (used by eval for latency)."""

    def __init__(self, judge: Judge):
        self.judge, self.name, self.times_ms = judge, judge.name, []

    def score(self, question: str, sentences: list[str]) -> float:
        t0 = time.perf_counter()
        try:
            return self.judge.score(question, sentences)
        finally:
            self.times_ms.append((time.perf_counter() - t0) * 1000)


def build_judge(name: str, model_dir: Path) -> Judge | None:
    if name in (None, "", "none"):
        return None
    if name in ("nli", "nli-t"):
        return NLIJudge(model_dir, template=name == "nli-t")
    if name in RERANKERS:
        return CrossEncoderJudge(model_dir, RERANKERS[name], name)
    raise ValueError(f"Unknown ANSWERABILITY_JUDGE {name!r} (use one of {JUDGES})")
