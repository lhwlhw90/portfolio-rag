"""Answer generation behind a pluggable interface.

- ExtractiveAnswerer (default): no model. Picks the most query-relevant
  sentences from retrieved chunks and tags each with a [n] citation marker.
  Its best-sentence score doubles as a confidence used for refusal.
- OpenAICompatAnswerer (opt-in): any OpenAI-compatible /v1/chat/completions
  endpoint (Ollama, llama.cpp server, vLLM, LM Studio). Optionally gated by
  the extractive confidence (refuse before calling the LLM) and falls back to
  extractive on any error so the demo never hard-fails.
"""
from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import dataclass, field
from typing import Iterator, Protocol

import httpx

from .retriever import Hit
from .text import content_tokens, identifiers, is_cjk, is_heading_like, split_sentences

log = logging.getLogger(__name__)

NO_ANSWER = "I couldn't find relevant information in the uploaded documents."
NO_ANSWER_ZH = "在已上传的文档中没有找到相关信息。"
_REFUSAL = re.compile(
    r"(couldn'?t find|could not find|can(?:no|')t find|cannot answer|can(?:no|')t answer|don'?t know|do not know|"
    r"not (?:mentioned|provided|specified|stated|included|available) in|no (?:relevant )?information|"
    r"(?:do|does) not (?:say|mention|contain|provide)|(?:don|doesn)'?t (?:say|mention|contain|provide)|"
    r"documents? do(?:es)? not say|文档中没有|文档未|没有找到|未找到|没有提到|未提及|没有相关|无法回答|不知道|没有提供|未提供|无相关)", re.I)


def no_answer_text(question: str) -> str:
    return NO_ANSWER_ZH if is_cjk(question) else NO_ANSWER


def looks_like_refusal(text: str) -> bool:
    return bool(_REFUSAL.search(text))


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
    refused: bool = False
    confidence: float | None = None      # extractive best-sentence score (also used as LLM gate)
    citation_mode: str = "explicit"      # explicit = answer has [n] markers; context = LLM gave none
    invalid_markers: list[int] = field(default_factory=list)  # [n] that don't map to a source
    latency_ms: float | None = None
    refusal_reason: str | None = None    # machine-readable code, see REFUSAL_REASONS
    refusal_detail: str | None = None    # human-readable explanation shown in the UI


REFUSAL_REASONS = {
    "no_context": "No documents matched the question.",
    "no_candidates": "The retrieved passages contain no usable sentences.",
    "low_confidence": "The best matching sentence is not similar enough to the question.",
    "unknown_identifier": "The question names a model / code / version that does not occur in the retrieved passages.",
    "llm_refused": "The language model said the sources do not answer the question.",
}


REFUSAL_REASONS_ZH = {
    "no_context": "没有检索到与问题相关的文档片段。",
    "no_candidates": "检索到的片段中没有可用的句子。",
    "low_confidence": "最相关的句子与问题的相似度不足。",
    "unknown_identifier": "问题中的型号 / 代码 / 版本号没有出现在检索到的片段中。",
    "llm_refused": "语言模型判断这些来源无法回答该问题。",
}


def refusal(question: str, backend: str, reason: str, detail: str | None = None, confidence: float | None = 0.0,
            latency_ms: float | None = None, text: str | None = None) -> Answer:
    base = (REFUSAL_REASONS_ZH if is_cjk(question) else REFUSAL_REASONS)[reason]
    return Answer(text or no_answer_text(question), [], backend, refused=True, confidence=confidence,
                  latency_ms=latency_ms, refusal_reason=reason, refusal_detail=f"{base} {detail}" if detail else base)


def best_supporting_sentence(answer: str, chunk_text: str) -> str | None:
    """The chunk sentence sharing the most content tokens with the answer, used as the
    citation snippet for LLM answers so the UI can highlight the exact evidence."""
    a = set(content_tokens(answer))
    best, best_ov = None, 0
    for sent in split_sentences(chunk_text):
        if is_heading_like(sent):
            continue
        ov = len(a & set(content_tokens(sent)))
        if ov > best_ov:
            best, best_ov = sent, ov
    return best


def missing_identifiers(question: str, context: str) -> list[str]:
    """Identifiers in the question (e.g. 'X3', 'E402', 'v2.4.0') with no counterpart in the
    context. Prefix matches count ('5G' ~ '5GHz') so that only clear mismatches trigger."""
    have = identifiers(context)
    return sorted(q for q in identifiers(question)
                  if not any(h.startswith(q) or q.startswith(h) for h in have))


def build_citations(hits: list[Hit], snippet_chars: int = 240) -> list[Citation]:
    cits = []
    for n, h in enumerate(hits, start=1):
        snip = h.chunk.text if len(h.chunk.text) <= snippet_chars else h.chunk.text[:snippet_chars].rstrip() + "…"
        cits.append(Citation(n, h.chunk.doc_id, h.chunk.doc_name, h.chunk.chunk_id, h.chunk.page, snip, round(h.score, 6)))
    return cits


def format_source(c: Citation) -> str:
    """Human-readable source label, e.g. 'handbook.pdf, p. 3'."""
    return f"{c.doc_name}, p. {c.page}" if c.page is not None else c.doc_name


def parse_markers(text: str) -> list[int]:
    """Extract [n], [1][2], [1, 2] and 【1】 style markers in order of appearance."""
    out: list[int] = []
    for grp in re.findall(r"[\[【]\s*(\d+(?:\s*[,，、]\s*\d+)*)\s*[\]】]", text):
        for n in re.split(r"\s*[,，、]\s*", grp):
            if int(n) not in out:
                out.append(int(n))
    return out


_SUBQ = re.compile(r"[?？；;]\s*|,\s+and\s+|，(?=[^，]*[？?]?$)|\s+and\s+(?=(?:what|which|who|when|where|how|is|are|can|do|does)\b)", re.I)


def split_subquestions(question: str) -> list[str]:
    """'How long is X, and when was Y released?' -> ['How long is X', 'when was Y released']."""
    parts = [p.strip() for p in _SUBQ.split(question) if p and p.strip()]
    return [p for p in parts if len(content_tokens(p)) >= 2] or [question]


def attribute_citations(text: str, pairs: list[tuple[Citation, str]], min_overlap: float = 0.5
                        ) -> tuple[str, list[int]]:
    """Post-hoc attribution: append [n] to each answer sentence whose content tokens are
    mostly (>= min_overlap) contained in source n. Returns (new_text, used_indices)."""
    src_tokens = [(c.index, set(content_tokens(t))) for c, t in pairs]
    out, used = [], []
    for sent in split_sentences(text) or [text]:
        toks = set(content_tokens(sent))
        best_idx, best = None, 0.0
        for idx, st in src_tokens:
            ov = len(toks & st) / (len(toks) or 1)
            if ov > best:
                best_idx, best = idx, ov
        if best_idx is not None and best >= min_overlap and len(toks) >= 3:  # skip echoed headings
            out.append(f"{sent} [{best_idx}]")
            if best_idx not in used:
                used.append(best_idx)
        else:
            out.append(sent)
    return " ".join(out), used


class Answerer(Protocol):
    name: str

    def answer(self, question: str, hits: list[Hit]) -> Answer: ...


class ExtractiveAnswerer:
    """Model-free answerer: selects the best sentences from retrieved chunks.

    Sentence score = 0.5 * IDF-weighted overlap with the question terms (stop-words removed)
    + cosine(sentence, question) if an embedder is given (handles paraphrases)
    + a small bonus for higher-ranked chunks. The best score is the answer's
    confidence; below `min_score` we refuse instead of guessing.
    """
    name = "extractive"

    def __init__(self, max_sentences: int = 3, embedder=None, keep_ratio: float = 0.85, min_score: float = 0.0,
                 lex_weight: float = 0.5, check_identifiers: bool = True):
        self.max_sentences, self.embedder, self.keep_ratio = max_sentences, embedder, keep_ratio
        self.min_score, self.lex_weight = min_score, lex_weight
        self.check_identifiers = check_identifiers

    def score_sentences(self, question: str, hits: list[Hit]) -> list[tuple[float, int, int, str, float]]:
        """Returns candidates (score, hit_rank, order, sentence, cosine), best first."""
        q = set(content_tokens(question))
        sents: list[tuple[int, int, str, set]] = []  # (hit_rank, order, sentence, tokens)
        for rank, h in enumerate(hits):
            for order, sent in enumerate(split_sentences(h.chunk.text)):
                toks = content_tokens(sent)
                if len(toks) < 3 or sent.endswith(("?", "？")):  # questions (e.g. FAQ headings) aren't answers
                    continue
                if is_heading_like(sent):  # title-like line ("星河 S7 智能扫地机器人 产品说明书")
                    continue
                sents.append((rank, order, sent, set(toks)))
        if not sents:
            return []
        # IDF over candidate sentences: query terms that appear everywhere (e.g. the product
        # name) carry little evidence; rare terms ("price", "CEO") must be covered to answer.
        n = len(sents)
        df = {t: sum(1 for s in sents if t in s[3]) for t in q}
        w = {t: math.log(1 + n / (1 + df[t])) for t in q}
        wsum = sum(w.values()) or 1.0
        if self.embedder is not None:
            sem = self.embedder.embed_documents([s[2] for s in sents]) @ self.embedder.embed_query(question)
        else:
            sem = [0.0] * len(sents)
        cands = []
        for (rank, order, sent, toks), cos in zip(sents, sem):
            lex = sum(w[t] for t in q & toks) / wsum
            if self.embedder is None and lex == 0:
                continue
            cands.append((self.lex_weight * lex + float(cos) + 0.05 / (rank + 1), rank, order, sent, float(cos)))
        cands.sort(key=lambda x: -x[0])
        return cands

    def confidence(self, cands) -> float:
        """Answerability signal. With an embedder: max sentence-question cosine (lexical
        overlap is deliberately excluded — eval showed it inflates scores for questions that
        merely mention a known entity, e.g. 'price of the RoboArm X2'). Without: best score."""
        if not cands:
            return 0.0
        return max(c[4] for c in cands) if self.embedder is not None else cands[0][0]

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        if not hits:
            return refusal(question, self.name, "no_context")
        cands = self.score_sentences(question, hits)
        if not cands:
            return refusal(question, self.name, "no_candidates")
        best, conf = cands[0][0], self.confidence(cands)
        if self.check_identifiers:
            missing = missing_identifiers(question, " ".join(h.chunk.text for h in hits))
            if missing:
                return refusal(question, self.name, "unknown_identifier", confidence=round(conf, 4),
                               detail=("未找到：" if is_cjk(question) else "Not found: ") + ", ".join(missing))
        if conf < self.min_score:
            return refusal(question, self.name, "low_confidence", confidence=round(conf, 4),
                           detail=(f"（{conf:.3f} < 阈值 {self.min_score:.3f}）" if is_cjk(question) else
                                   f"(similarity {conf:.3f} < threshold {self.min_score:.3f})"))
        picked = [c for c in cands if c[0] >= best * self.keep_ratio][: self.max_sentences]
        subs = split_subquestions(question)
        if len(subs) > 1:  # multi-part question: make sure every part gets its best sentence
            for sq in subs:
                sc = self.score_sentences(sq, hits)
                if sc and self.confidence(sc) >= self.min_score and all(sc[0][3] != c[3] for c in picked):
                    picked.append(sc[0])
            picked = picked[: self.max_sentences + len(subs)]
        picked.sort(key=lambda c: (c[1], c[2]))  # keep reading order
        text = " ".join(f"{c[3]} [{c[1] + 1}]" for c in picked)
        support: dict[int, list[str]] = {}
        for _, rank, _, sent, _ in picked:
            support.setdefault(rank + 1, []).append(sent)
        used = [c for c in build_citations(hits) if c.index in support]
        for c in used:  # snippet = the exact supporting sentence(s), so the citation is verifiable
            c.snippet = " … ".join(support[c.index])
        return Answer(text, used, self.name, confidence=round(conf, 4))


# Prompt tuned on qwen2.5:0.5b: "[1] (file)" style headers were echoed verbatim by the small
# model, XML-tagged sources were not (see docs/DESIGN.md, LLM section).
SYSTEM_PROMPT = (
    "You answer questions about company documents. Use only the information inside the <source> tags. "
    "Write one or two short sentences and put the id of the source you used in square brackets at the end "
    "of each sentence, for example: The warranty is 2 years [3]. If the sources do not answer the question, "
    "reply: The documents do not say. Answer in the same language as the question.")


class OpenAICompatAnswerer:
    name = "openai"

    def __init__(self, base_url: str, model: str, api_key: str = "not-needed", timeout: float = 60.0,
                 fallback: Answerer | None = None, client: httpx.Client | None = None,
                 gate: ExtractiveAnswerer | None = None, max_tokens: int = 256):
        self.base_url, self.model, self.api_key = base_url.rstrip("/"), model, api_key
        self.fallback = fallback or ExtractiveAnswerer()
        self.gate = gate  # if set: refuse without calling the LLM when retrieval confidence is too low
        self.max_tokens = max_tokens
        self.client = client or httpx.Client(timeout=timeout)

    @staticmethod
    def build_prompt(question: str, pairs: list[tuple[Citation, str]]) -> str:
        ctx = "\n".join(f'<source id="{c.index}" file="{format_source(c)}">\n{text}\n</source>' for c, text in pairs)
        return f"{ctx}\n\nQuestion: {question}"

    def _prepare(self, question: str, hits: list[Hit]):
        """Returns (early_answer, citations, (citation, text) pairs, gate confidence)."""
        if not hits:
            return refusal(question, self.name, "no_context", confidence=None), [], [], None
        confidence = None
        if self.gate is not None:
            g = self.gate.answer(question, hits)
            confidence = g.confidence
            if g.refused:
                g.backend = f"{self.name}(gated)"
                return g, [], [], confidence
        cits = build_citations(hits)
        return None, cits, [(c, h.chunk.text) for c, h in zip(cits, hits)], confidence

    def _payload(self, question: str, pairs, stream: bool = False) -> dict:
        return {"model": self.model, "temperature": 0.0, "max_tokens": self.max_tokens, "stream": stream,
                "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                             {"role": "user", "content": self.build_prompt(question, pairs)}]}

    def _fallback(self, question: str, hits: list[Hit], err: Exception) -> Answer:
        log.warning("LLM backend failed (%s); falling back to extractive", err)
        ans = self.fallback.answer(question, hits)
        ans.backend = f"{self.name}->fallback:{ans.backend}"
        return ans

    def _finalize(self, question: str, text: str, cits, pairs, confidence, latency: float) -> Answer:
        text = re.sub(r"(\[\d+\])(\s*\1)+", r"\1", text.strip())  # "[3] [3]" -> "[3]"
        markers = parse_markers(text)
        valid = {c.index for c in cits}
        invalid = [m for m in markers if m not in valid]
        if looks_like_refusal(text) and not [m for m in markers if m in valid]:
            return refusal(question, self.name, "llm_refused", confidence=confidence, latency_ms=latency, text=text)
        full = {c.index: t for c, t in pairs}

        def sharpen(used):  # snippet = the chunk sentence that best supports the answer
            for c in used:
                c.snippet = best_supporting_sentence(text, full[c.index]) or c.snippet
            return used
        used = [c for c in cits if c.index in markers]
        if used:
            return Answer(text, sharpen(used), self.name, confidence=confidence, invalid_markers=invalid,
                          latency_ms=latency)
        # Small models often forget markers: attribute each answer sentence to the source it overlaps most.
        text2, attributed = attribute_citations(text, pairs)
        if attributed:
            return Answer(text2, sharpen([c for c in cits if c.index in attributed]), self.name, confidence=confidence,
                          citation_mode="attributed", invalid_markers=invalid, latency_ms=latency)
        # nothing attributable: expose retrieved sources as context, flagged as such
        return Answer(text, cits, self.name, confidence=confidence, citation_mode="context",
                      invalid_markers=invalid, latency_ms=latency)

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        t0 = time.perf_counter()
        early, cits, pairs, confidence = self._prepare(question, hits)
        if early is not None:
            return early
        try:
            r = self.client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}"},
                                 json=self._payload(question, pairs))
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
        except Exception as e:  # network / model errors -> graceful fallback
            return self._fallback(question, hits, e)
        return self._finalize(question, text, cits, pairs, confidence, (time.perf_counter() - t0) * 1000)

    def stream(self, question: str, hits: list[Hit]) -> Iterator[tuple[str, object]]:
        """Yields ("token", str) as the model generates, then ("answer", Answer) after the same
        post-processing as answer(). A ("reset", None) event means: discard streamed tokens."""
        t0 = time.perf_counter()
        early, cits, pairs, confidence = self._prepare(question, hits)
        if early is not None:
            yield "answer", early
            return
        parts: list[str] = []
        try:
            with self.client.stream("POST", f"{self.base_url}/chat/completions",
                                    headers={"Authorization": f"Bearer {self.api_key}"},
                                    json=self._payload(question, pairs, stream=True)) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    delta = (json.loads(data).get("choices") or [{}])[0].get("delta", {}).get("content")
                    if delta:
                        parts.append(delta)
                        yield "token", delta
        except Exception as e:
            if parts:
                yield "reset", None
            yield "answer", self._fallback(question, hits, e)
            return
        yield "answer", self._finalize(question, "".join(parts), cits, pairs, confidence,
                                       (time.perf_counter() - t0) * 1000)


def stream_answer(answerer, question: str, hits: list[Hit]) -> Iterator[tuple[str, object]]:
    """Uniform streaming interface: native token streaming if the answerer supports it,
    otherwise the finished answer is emitted sentence by sentence."""
    if hasattr(answerer, "stream"):
        yield from answerer.stream(question, hits)
        return
    ans = answerer.answer(question, hits)
    if not ans.refused:
        for piece in re.findall(r".+?(?:\[\d+\]\s*|$)", ans.answer):
            if piece:
                yield "token", piece
    yield "answer", ans


def build_answerer(settings, embedder=None) -> Answerer:
    extractive = ExtractiveAnswerer(embedder=embedder, min_score=settings.resolved_min_score)
    if settings.llm_backend == "extractive":
        return extractive
    if settings.llm_backend == "openai":
        return OpenAICompatAnswerer(settings.llm_base_url, settings.llm_model, settings.llm_api_key,
                                    settings.llm_timeout, fallback=extractive,
                                    gate=extractive if settings.llm_gate else None)
    raise ValueError(f"Unknown LLM_BACKEND '{settings.llm_backend}' (use 'extractive' or 'openai')")
