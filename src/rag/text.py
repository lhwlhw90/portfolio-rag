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


def _blocks(text: str) -> list[tuple[str, bool]]:
    """(block, is_heading) pairs. Single newlines are joined (hard-wrapped Markdown / PDF text),
    blank lines start a new block, '#' lines and short title-like lines are heading blocks."""
    blocks: list[tuple[str, bool]] = []
    buf = ""
    for line in text.split("\n"):
        line = re.sub(r"^(?:[-*•]|\d+[.)、])\s+", "", line.strip())  # drop list bullets
        if not line:
            if buf:
                blocks.append((buf, False))
                buf = ""
            continue
        if line.startswith("#"):
            if buf:
                blocks.append((buf, False))
                buf = ""
            blocks.append((line.lstrip("#").strip(), True))
            continue
        if not buf and is_heading_like(line):
            blocks.append((line, True))
            continue
        buf = _join(buf, line) if buf else line
    if buf:
        blocks.append((buf, False))
    return blocks


def _sents(block: str) -> list[str]:
    return [s.strip() for s in _SENT_END.split(block) if s and s.strip()]


def split_sentences(text: str) -> list[str]:
    """Sentence splitter that tolerates hard-wrapped lines (Markdown, PDF text):
    single newlines are joined, blank lines / headings start a new block.
    Handles both '. ' (English) and '。！？；' (Chinese, no trailing space)."""
    return [s for b, _ in _blocks(text) for s in _sents(b)]


def _usable_heading(h: str) -> bool:
    return bool(h) and not h.endswith(("?", "？")) and len(h) <= 80


def sentences_with_headings(text: str, inherited: str | None = None) -> list[tuple[str, str | None]]:
    """Like split_sentences, but each sentence is paired with the nearest heading above it
    (`inherited` = heading in force where the chunk starts, for chunks that begin mid-section).
    Order and content of the sentences are identical to split_sentences(text)."""
    out: list[tuple[str, str | None]] = []
    cur = inherited
    for n, (b, is_head) in enumerate(_blocks(text)):
        for s in _sents(b):
            out.append((s, None if is_head else cur))
        if n == 0 and is_head and inherited and inherited.endswith(b):
            continue  # the chunk starts inside its (inherited) heading line: keep the full heading
        if is_head and _usable_heading(b):
            cur = b
    return out


def last_heading(text: str, inherited: str | None = None) -> str | None:
    """Heading in force at the end of `text` (carried into the next chunk / page)."""
    cur = inherited
    for b, is_head in _blocks(text):
        if is_head and _usable_heading(b):
            cur = b
    return cur


def is_terminal(sentence: str) -> bool:
    return bool(_TERMINAL.search(sentence))


def is_heading_like(line: str) -> bool:
    """Title/heading lines: no sentence-final punctuation and short (few words, or <= 24 chars
    for CJK where bigram counts overstate length)."""
    return not is_terminal(line) and (len(tokenize(line)) < 8 or (is_cjk(line) and len(line) <= 24))


_IDENT = re.compile(r"(?<![A-Za-z0-9.])[A-Za-z]*\d[A-Za-z0-9]*(?:[.\-][A-Za-z0-9]+)*")


def identifiers(text: str) -> set[str]:
    """Model numbers, error codes, versions: alphanumeric tokens that contain a digit and at
    least one letter or dot ("X3", "E402", "v2.4.0", "S9", "2.4"). Plain integers are
    excluded: questions like "after 5 failed attempts" legitimately use numbers that are
    phrased differently in the source ("five")."""
    out = set()
    for m in _IDENT.findall(text):
        m = m.lower().rstrip(".-")
        if any(c.isdigit() for c in m) and (any(c.isalpha() for c in m) or "." in m):
            out.add(m)
    return out
