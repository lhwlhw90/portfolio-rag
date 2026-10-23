"""Query expansion for paraphrased questions ("Somebody stole my work notebook computer"
vs. a document that says "lost or stolen laptop").

Modes (QUERY_EXPANSION):
- ``off``      original query only (M3 behaviour).
- ``prf``      model-free pseudo-relevance feedback: add the most distinctive terms of the
               top first-pass chunks (RM3-style) as a second query.
- ``rewrite``  a local LLM (OpenAI-compatible, e.g. Ollama qwen2.5) rewrites the question into
               a few search queries in document vocabulary.
- ``hyde``     HyDE: the LLM writes a short hypothetical answer passage, which is used as an
               additional query (its wording is closer to the documents than the question's).

The retriever fuses the ranked lists of the original query and the expansions with weighted
RRF (original weight 1, expansions share EXPANSION_WEIGHT). The *answerer* always sees the
original question. LLM failures/timeouts degrade to no expansion. The default mode is chosen on
the eval dev split (see eval/results/REPORT.md section 6).
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from collections import Counter
from pathlib import Path

import httpx

from .text import content_tokens, is_cjk

log = logging.getLogger(__name__)
MODES = ("off", "prf", "rewrite", "hyde")

PROMPTS = {
    ("rewrite", "en"): ("Rewrite the user's question as 3 short search queries for a company knowledge base "
                        "(HR policies, IT helpdesk, product manuals). Use the formal vocabulary such documents "
                        "use and replace slang or paraphrases with standard terms. Output one query per line, "
                        "no numbering, no explanations."),
    ("rewrite", "zh"): ("把用户的问题改写成3条简短的检索查询，用于检索公司知识库（人事制度、IT服务、产品说明书）。"
                        "使用这类文档中的正式用语，把口语或同义说法换成标准术语。每行一条，不要编号，不要解释。"),
    ("hyde", "en"): ("Write a short passage (two sentences) as it might appear in a company document "
                     "(HR policy, IT helpdesk guide or product manual) that answers the question. "
                     "Invent plausible details if needed. Output only the passage."),
    ("hyde", "zh"): ("写一段简短的文字（两句话），像是公司文档（人事制度、IT服务指南或产品说明书）中回答该问题的段落。"
                     "必要时可以编造合理的细节。只输出这段文字。"),
}


def prf_terms(query: str, texts: list[str], all_texts: list[str], n_terms: int = 6) -> list[str]:
    """Top tf-idf terms of the feedback texts that are not already in the query."""
    q = set(content_tokens(query))
    df = Counter(t for doc in all_texts for t in set(content_tokens(doc)))
    n = max(len(all_texts), 1)
    tf = Counter(t for doc in texts for t in content_tokens(doc))
    scored = {t: c * math.log(1 + n / df.get(t, 1)) for t, c in tf.items()
              if t not in q and not t.isdigit() and df.get(t, 0) < n}
    return [t for t, _ in sorted(scored.items(), key=lambda x: (-x[1], x[0]))[:n_terms]]


class QueryExpander:
    def __init__(self, mode: str = "off", base_url: str = "http://localhost:11434/v1", model: str = "qwen2.5:1.5b",
                 timeout: float = 20.0, weight: float = 1.0, prf_docs: int = 3, client: httpx.Client | None = None,
                 cache_path: Path | None = None, cooldown: float = 60.0):
        if mode not in MODES:
            raise ValueError(f"unknown QUERY_EXPANSION {mode!r} (use one of {MODES})")
        self.mode, self.base_url, self.model, self.weight, self.prf_docs = mode, base_url.rstrip("/"), model, weight, prf_docs
        self.client = client or httpx.Client(timeout=timeout)
        # (mode, model, question) -> (expansions, ms it took to compute them). Optionally persisted
        # as JSON (the eval uses eval/results/qe_cache_*.json so reruns don't repeat LLM calls).
        self.cache: dict[tuple[str, str, str], tuple[list[str], float]] = {}
        self.cache_path = cache_path
        if cache_path is not None and Path(cache_path).exists():
            for row in json.loads(Path(cache_path).read_text(encoding="utf-8")):
                self.cache[(row["mode"], row["model"], row["question"])] = (row["expansions"], row["ms"])
        self.cooldown, self._down_until = cooldown, 0.0  # after an LLM failure, skip expansion for `cooldown` s
        self.last_ms = 0.0    # wall time of the last expand() call
        self.last_cost_ms = 0.0  # time the expansion took when it was computed (== last_ms unless cached)

    def _llm(self, question: str) -> list[str]:
        lang = "zh" if is_cjk(question) else "en"
        r = self.client.post(f"{self.base_url}/chat/completions", json={
            "model": self.model, "temperature": 0.0, "max_tokens": 120, "stream": False,
            "messages": [{"role": "system", "content": PROMPTS[(self.mode, lang)]},
                         {"role": "user", "content": question}]})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"].strip()
        if self.mode == "hyde":
            return [text] if text else []
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)、])\s*", "", ln).strip().strip('"') for ln in text.splitlines()]
        return [ln for ln in lines if ln and ln.lower() != question.lower()][:3]

    def expand(self, question: str, retriever=None) -> list[str]:
        """Extra queries for `question` (empty list = no expansion). `retriever` is needed for prf."""
        t0 = time.perf_counter()
        cached = None
        try:
            if self.mode == "off":
                return []
            key = (self.mode, self.model if self.mode != "prf" else "", question)
            if key in self.cache:
                cached = self.cache[key][1]
                return self.cache[key][0]
            if self.mode == "prf":
                if retriever is None or not retriever.store.chunks:
                    return []
                hits = retriever.retrieve(question, k=self.prf_docs, mode="hybrid")
                terms = prf_terms(question, [h.chunk.text for h in hits], [c.text for c in retriever.store.chunks])
                out = [" ".join(terms)] if terms else []  # space-separated: CJK bigrams stay intact
            else:
                if time.monotonic() < self._down_until:
                    return []
                try:
                    out = self._llm(question)
                except Exception as e:  # unreachable server, timeout, bad JSON -> no expansion
                    log.warning("query expansion (%s) failed: %s; disabled for %.0f s", self.mode, e, self.cooldown)
                    self._down_until = time.monotonic() + self.cooldown
                    return []
            self.cache[key] = (out, round((time.perf_counter() - t0) * 1000, 1))
            return out
        finally:
            self.last_ms = (time.perf_counter() - t0) * 1000
            self.last_cost_ms = cached if cached is not None else self.last_ms

    def save_cache(self) -> None:
        if self.cache_path is None:
            return
        rows = [{"mode": m, "model": mo, "question": q, "expansions": e, "ms": ms}
                for (m, mo, q), (e, ms) in sorted(self.cache.items()) if m != "prf"]
        old = json.loads(Path(self.cache_path).read_text(encoding="utf-8")) if Path(self.cache_path).exists() else []
        keep = [r for r in old if (r["mode"], r["model"], r["question"]) not in self.cache]
        Path(self.cache_path).write_text(json.dumps(keep + rows, ensure_ascii=False, indent=0), encoding="utf-8")
