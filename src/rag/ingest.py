"""Document parsing (txt / md / pdf) and chunking with metadata."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

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


def parse_document(filename: str, data: bytes) -> list[Page]:
    ext = ext_of(filename)
    if ext not in SUPPORTED_EXTS:
        raise UnsupportedFileType(f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTS)}")
    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(data))
        return [Page(text=(p.extract_text() or ""), page=i + 1) for i, p in enumerate(reader.pages)]
    text = data.decode("utf-8", errors="replace")
    return [Page(text=text, page=None)]


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
    chunks: list[Chunk] = []
    for page in pages:
        for start, piece in split_text(page.text, chunk_size, overlap):
            chunks.append(Chunk(doc_id=doc_id, doc_name=doc_name, chunk_index=len(chunks),
                                text=piece, page=page.page, start_char=start))
    return chunks
