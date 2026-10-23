"""Answer generation behind a pluggable interface.

- ExtractiveAnswerer (default): no model. Picks the most query-relevant
  sentences from retrieved chunks and tags each with a [n] citation marker.
- OpenAICompatAnswerer (opt-in): any OpenAI-compatible /v1/chat/completions
  endpoint (Ollama, llama.cpp server, vLLM, LM Studio). Falls back to
  extractive on any error so the demo never hard-fails.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Protocol

import httpx

from .embeddings import tokenize
from .retriever import Hit

log = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find relevant information in the uploaded documents."
_STOP = set("a an the of to in on for and or is are was were be been by with as at from that this it its "
            "what which who whom how why when where do does did can could should would will shall i you we "
            "they he she them his her our your their there here about into than then so not no yes".split())


@dataclass
class Citation:
    index: int            # the [n] marker used in the answer
    doc_id: str
    doc_name: str
    chunk_id: str
    page: int | None
    snippet: str
    score: float


@dataclass
class Answer:
    answer: str
    citations: list[Citation] = field(default_factory=list)
    backend: str = "extractive"


def build_citations(hits: list[Hit], snippet_chars: int = 240) -> list[Citation]:
    cits = []
    for n, h in enumerate(hits, start=1):
        snip = h.chunk.text if len(h.chunk.text) <= snippet_chars else h.chunk.text[:snippet_chars].rstrip() + "…"
        cits.append(Citation(n, h.chunk.doc_id, h.chunk.doc_name, h.chunk.chunk_id, h.chunk.page, snip, round(h.score, 6)))
    return cits


def format_source(c: Citation) -> str:
    """Human-readable source label, e.g. 'handbook.pdf, p. 3'."""
    return f"{c.doc_name}, p. {c.page}" if c.page is not None else c.doc_name


class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: list[Hit]) -> Answer: ...


_SENT_END = re.compile(r"(?<=[.!?。！？])\s+")
_TERMINAL = re.compile(r"[.!?。！？:;]$")


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that tolerates hard-wrapped lines (Markdown, PDF text):
    single newlines are joined, blank lines / headings start a new block."""
    blocks: list[str] = []
    buf: list[str] = []

    def flush():
        if buf:
            blocks.append(" ".join(buf))
            buf.clear()

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            blocks.append(line.lstrip("#").strip())
            continue
        if not buf and not _TERMINAL.search(line) and len(tokenize(line)) < 8:  # heading-like
            blocks.append(line)
            continue
        buf.append(line)
    flush()
    return [s.strip() for b in blocks for s in _SENT_END.split(b) if s.strip()]


class ExtractiveAnswerer:
    """Model-free answerer: selects the best sentences from retrieved chunks.

    Sentence score = lexical overlap with the question (stop-words removed)
    plus, if an embedder is given, cosine similarity between the sentence and
    the question (handles paraphrases), plus a small bonus for higher-ranked chunks.
    """
    name = "extractive"

    def __init__(self, max_sentences: int = 3, embedder=None, keep_ratio: float = 0.85, min_score: float = 0.0):
        self.max_sentences, self.embedder, self.keep_ratio = max_sentences, embedder, keep_ratio
        self.min_score = min_score  # below this best-sentence score we answer "not found"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(NO_ANSWER, [], self.name)
        cits = build_citations(hits)
        q = {t for t in tokenize(question) if t not in _STOP}
        sents: list[tuple[int, int, str, float]] = []  # (hit_rank, order, sentence, lexical)
        for rank, h in enumerate(hits):
            for order, sent in enumerate(split_sentences(h.chunk.text)):
                toks = [t for t in tokenize(sent) if t not in _STOP]
                if len(toks) < 3 or sent.endswith("?"):  # questions (e.g. FAQ headings) aren't answers
                    continue
                if not _TERMINAL.search(sent) and len(toks) < 8:  # heading/title-like line
                    continue
                sents.append((rank, order, sent, len(q & set(toks)) / (len(q) or 1)))
        if not sents:
            return Answer(NO_ANSWER, [], self.name)
        if self.embedder is not None:
            sem = self.embedder.embed_documents([s[2] for s in sents]) @ self.embedder.embed_query(question)
        else:
            sem = [0.0] * len(sents)
        cands = []
        for (rank, order, sent, lex), cos in zip(sents, sem):
            if self.embedder is None and lex == 0:
                continue
            cands.append((0.5 * lex + float(cos) + 0.05 / (rank + 1), rank, order, sent))
        if not cands:
            return Answer(NO_ANSWER, [], self.name)
        cands.sort(key=lambda x: -x[0])
        best = cands[0][0]
        if best < self.min_score:
            return Answer(NO_ANSWER, [], self.name)
        picked = [c for c in cands if c[0] >= best * self.keep_ratio][: self.max_sentences]
        picked.sort(key=lambda c: (c[1], c[2]))  # keep reading order
        text = " ".join(f"{s} [{rank + 1}]" for _, rank, _, s in picked)
        support: dict[int, list[str]] = {}
        for _, rank, _, sent in picked:
            support.setdefault(rank + 1, []).append(sent)
        used = [c for c in cits if c.index in support]
        for c in used:  # snippet = the exact supporting sentence(s), so the citation is verifiable
            c.snippet = " … ".join(support[c.index])
        return Answer(text, used, self.name)


SYSTEM_PROMPT = (
    "You answer questions using ONLY the provided sources. Cite every claim with the source number "
    "in square brackets, e.g. [1] or [2][3]. If the sources do not contain the answer, say you don't know. "
    "Be concise.")


class OpenAICompatAnswerer:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", timeout: float = 60.0,
                 fallback: Answerer | None = None, client: httpx.Client | None = None):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.fallback = fallback or ExtractiveAnswerer()
        self.client = client or httpx.Client(timeout=timeout)

    def _prompt(self, question: str, cits: list[Citation]) -> str:
        ctx = "\n\n".join(f"[{c.index}] ({format_source(c)})\n{h}" for c, h in cits)
        return f"Sources:\n{ctx}\n\nQuestion: {question}\nAnswer with citations:"

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return Answer(NO_ANSWER, [], self.name)
        cits = build_citations(hits)
        pairs = [(c, h.chunk.text) for c, h in zip(cits, hits)]
        try:
            r = self.client.post(f"{self.base_url}/chat/completions",
                                 headers={"Authorization": f"Bearer {self.api_key}"},
                                 json={"model": self.model, "temperature": 0.1, "messages": [
                                     {"role": "system", "content": SYSTEM_PROMPT},
                                     {"role": "user", "content": self._prompt(question, pairs)}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:  # network / model errors -> graceful fallback
            log.warning("LLM backend failed (%s); falling back to extractive", e)
            ans = self.fallback.answer(question, hits)
            ans.backend = f"{self.name}->fallback:{ans.backend}"
            return ans
        used = {int(m) for m in re.findall(r"\[(\d+)\]", text)}
        return Answer(text, [c for c in cits if c.index in used] or cits, self.name)


def build_answerer(settings, embedder=None) -> Answerer:
    extractive = ExtractiveAnswerer(embedder=embedder, min_score=settings.answer_min_score)
    if settings.llm_backend == "extractive":
        return extractive
    if settings.llm_backend == "openai":
        return OpenAICompatAnswerer(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                                    settings.llm_timeout, fallback=extractive)
    raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}' (use 'extractive' or 'openai')")
