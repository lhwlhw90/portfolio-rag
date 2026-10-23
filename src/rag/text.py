"""Language-aware text utilities shared by BM25, the hash embedder and the
extractive answerer. English: lowercase words + light plural stemming.
Chinese (CJK): character bigrams — dependency-free, robust word-segmentation
substitute that works well for BM25 on short documents."""
from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

STOPWORDS = set(
    "a an the of to in on for and or is are was were be been by with as at from that this it its "
    "what which who whom how why when where do does did can could should would will shall i you we "
    "they he she them his her our your their there here about into than then so not no yes much many".split()
) | {  # common Chinese function/question bigrams
    "什么", "怎么", "怎样", "多少", "哪些", "哪个", "是否", "可以", "我们", "你们", "他们", "如何",
    "的是", "是什", "么时", "时候", "需要", "一下", "请问", "这个", "那个", "有没", "没有", "吗",
    "的", "了", "是", "在", "和", "与", "或", "吧", "呢", "啊",
}


def _stem(tok: str) -> str:
    """Tiny suffix stripper (plurals) so 'passwords' matches 'password'."""
    if len(tok) > 4 and tok.endswith("ies"):
        return tok[:-3] + "y"
    if len(tok) > 3 and tok.endswith("s") and not tok.endswith(("ss", "us", "is")):
        return tok[:-1]
    return tok


def tokenize(text: str) -> list[str]:
    out: list[str] = []
    for run in _TOKEN.findall(text.lower()):
        if _CJK.match(run):
            out.extend([run] if len(run) == 1 else [run[i:i + 2] for i in range(len(run) - 1)])
        else:
            out.append(_stem(run))
            parts = re.findall(r"[a-z]+|[0-9]+", run)
            if len(parts) > 1:  # 'ip54' also yields 'ip', '54'; 'x2' -> 'x', '2'
                out.extend(p for p in parts if len(p) > 1)
    return out


def content_tokens(text: str) -> list[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


def is_cjk(text: str) -> bool:
    return bool(_CJK.search(text))


_SENT_END = re.compile(r"(?<=[.!?])\s+|(?<=[。！？；])\s*")
_TERMINAL = re.compile(r"[.!?。！？:;：；]$")


def _join(a: str, b: str) -> str:
    return a + b if (a and b and _CJK.match(a[-1]) and _CJK.match(b[0])) else f"{a} {b}"


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that tolerates hard-wrapped lines (Markdown, PDF text):
    single newlines are joined, blank lines / headings start a new block.
    Handles both '. ' (English) and '。！？；' (Chinese, no trailing space)."""
    blocks: list[str] = []
    buf = ""
    for line in text.split("\n"):
        line = re.sub(r"^(?:[-*•]|\d+[.)、])\s+", "", line.strip())  # drop list bullets
        if not line:
            if buf:
                blocks.append(buf)
                buf = ""
            continue
        if line.startswith("#"):
            if buf:
                blocks.append(buf)
                buf = ""
            blocks.append(line.lstrip("#").strip())
            continue
        if not buf and is_heading_like(line):
            blocks.append(line)
            continue
        buf = _join(buf, line) if buf else line
    if buf:
        blocks.append(buf)
    return [s.strip() for b in blocks for s in _SENT_END.split(b) if s and s.strip()]


def is_terminal(sentence: str) -> bool:
    return bool(_TERMINAL.search(sentence))


def is_heading_like(line: str) -> bool:
    """Title/heading lines: no sentence-final punctuation and short (few words, or <= 24 chars
    for CJK where bigram counts overstate length)."""
    return not is_terminal(line) and (len(tokenize(line)) < 8 or (is_cjk(line) and len(line) <= 24))
