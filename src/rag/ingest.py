"""Document parsing (txt / md / pdf) and chunking with metadata."""
from __future__ import annotations

import io
import re
import unicodedata
from dataclasses import dataclass, field

from .text import last_heading

SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf"}


@dataclass
class Page:
    text: str
    page: int | None  # 1-based page number for PDFs, None for plain text


@dataclass
class Chunk:
    doc_id: str
    doc_name: str
    chunk_index: int
    text: str
    page: int | None
    start_char: int
    meta: dict = field(default_factory=dict)

    @property
    def chunk_id(self) -> str:
        return f"{self.doc_id}:{self.chunk_index}"


class UnsupportedFileType(ValueError):
    pass


def ext_of(filename: str) -> str:
    m = re.search(r"(\.[A-Za-z0-9]+)$", filename)
    return m.group(1).lower() if m else ""


# PDF fonts (e.g. Noto CJK via Chrome/Skia) often map glyphs to *radical* code points in their
# ToUnicode tables: "⼊⻔" instead of "入门". Fold them back to the unified ideographs, otherwise
# BM25/embeddings and citation matching silently break on Chinese PDFs.
_RADICAL_SUPPLEMENT = dict(zip(
    "⺰⻅⻆⻈⻉⻋⻐⻓⻔⻙⻚⻛⻜⻠⻢⻥⻦⻧⻨⻩⻪⻬⻮⻰⻳⺠⺟",
    "纟见角讠贝车钅长门韦页风飞饣马鱼鸟卤麦黄黾齐齿龙龟民母"))


def fold_cjk_radicals(text: str) -> str:
    out = []
    for ch in text:
        o = ord(ch)
        if 0x2F00 <= o <= 0x2FDF or 0xF900 <= o <= 0xFAFF:  # Kangxi radicals, compatibility ideographs
            out.append(unicodedata.normalize("NFKC", ch))
        else:
            out.append(_RADICAL_SUPPLEMENT.get(ch, ch))
    return "".join(out)


def parse_document(filename: str, data: bytes) -> list[Page]:
    ext = ext_of(filename)
    if ext not in SUPPORTED_EXTS:
        raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [Page(text=fold_cjk_radicals(p.extract_text() or ""), page=i + 1) for i, p in enumerate(reader.pages)]
    text = data.decode("utf-8-sig", errors="replace")
    return [Page(text=fold_cjk_radicals(text), page=None)]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, chunk_size: int, overlap: int) -> list[tuple[int, str]]:
    """Split into ~chunk_size char windows, preferring paragraph/sentence/word
    boundaries. Returns (start_offset, chunk_text) pairs."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if not 0 <= overlap < chunk_size:
        raise ValueError("overlap must be in [0, chunk_size)")
    text = _normalize(text)
    if not text:
        return []
    out: list[tuple[int, str]] = []
    start, n = 0, len(text)
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            window = text[start:end]
            cut = -1
            for sep in ("\n\n", "\n", ". ", "? ", "! ", "。", "！", "？", "；", " "):
                idx = window.rfind(sep)
                if idx > chunk_size * 0.5:
                    cut = idx + len(sep)
                    break
            if cut > 0:
                end = start + cut
        piece = text[start:end].strip()
        if piece:
            out.append((start, piece))
        if end >= n:
            break
        nxt = end - overlap
        # snap overlap start to a sentence boundary if one is inside the overlap, else a word boundary
        sent = max(text.rfind(". ", nxt, end), text.rfind("\n", nxt, end), text.rfind("。", nxt, end))
        if sent != -1 and sent + 1 < end:
            start = sent + 1
            while start < end and text[start] in " \n":
                start += 1
        else:
            sp = text.find(" ", nxt, end)
            start = sp + 1 if sp != -1 else nxt
        prev_start = out[-1][0] if out else -1
        if start <= prev_start:  # guarantee forward progress
            start = end
    return out


def chunk_pages(doc_id: str, doc_name: str, pages: list[Page], chunk_size: int, overlap: int) -> list[Chunk]:
    """Chunks carry meta["heading"]: the section heading in force where the chunk starts (also
    across page breaks), so answers from mid-section chunks can still show e.g. the version /
    date that only appears in the heading."""
    chunks: list[Chunk] = []
    carried: str | None = None
    for page in pages:
        norm = _normalize(page.text)
        for start, piece in split_text(page.text, chunk_size, overlap):
            line_start = norm.rfind("\n", 0, start) + 1
            heading = last_heading(norm[:line_start], carried)  # whole lines before the chunk
            if line_start < start:  # chunk starts mid-line: if that line is a heading, it is the full one
                line_end = norm.find("\n", start)
                heading = last_heading(norm[line_start: line_end if line_end != -1 else len(norm)], heading)
            chunks.append(Chunk(doc_id=doc_id, doc_name=doc_name, chunk_index=len(chunks),
                                text=piece, page=page.page, start_char=start,
                                meta={"heading": heading} if heading else {}))
        carried = last_heading(norm, carried)
    return chunks
