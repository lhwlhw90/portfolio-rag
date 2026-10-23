"""Server-side PDF page rendering with highlighted citation snippets (pypdfium2 + Pillow).

The browser gets a plain PNG, so the demo needs no pdf.js bundle and works offline.
Highlights are found by searching the snippet in pdfium's per-character text layer,
ignoring whitespace (our chunks join hard-wrapped lines; the PDF keeps the line breaks).
"""
from __future__ import annotations

import io
import re

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_c
from PIL import Image, ImageDraw

from .ingest import fold_cjk_radicals

HIGHLIGHT = (255, 214, 0, 110)
MAX_SCALE = 3.0


def page_count(data: bytes) -> int:
    pdf = pdfium.PdfDocument(data)
    try:
        return len(pdf)
    finally:
        pdf.close()


def _snippet_parts(snippet: str) -> list[str]:
    """Citation snippets join sentences with ' … ' and may end with a truncation '…'."""
    parts = [re.sub(r"\s+", "", p) for p in re.split(r"…|\.\.\.", snippet)]
    return [p for p in parts if len(p) >= 4]


def find_rects(page, needles: list[str]) -> list[tuple[float, float, float, float]]:
    """Return highlight rectangles in PDF points (left, bottom, right, top), one per text line."""
    tp = page.get_textpage()
    try:
        n = tp.count_chars()
        chars, index = [], []  # whitespace-free page text + map back to pdfium char indices
        for i in range(n):
            ch = fold_cjk_radicals(chr(pdfium_c.FPDFText_GetUnicode(tp.raw, i)))
            if not ch.isspace() and ch not in "\x00\ufffe\uffff":
                chars.append(ch.lower())
                index.append(i)
        hay = "".join(chars)
        rects = []
        for needle in needles:
            for part in _snippet_parts(needle):
                pos = hay.find(part.lower())
                if pos < 0:
                    continue
                line: list[float] | None = None
                for j in range(pos, pos + len(part)):
                    l, b, r, t = tp.get_charbox(index[j])
                    if line and abs(b - line[1]) < 0.5 * (t - b + 1e-6) and l >= line[0] - 1:
                        line = [min(line[0], l), min(line[1], b), max(line[2], r), max(line[3], t)]
                    else:
                        if line:
                            rects.append(tuple(line))
                        line = [l, b, r, t]
                if line:
                    rects.append(tuple(line))
        return rects
    finally:
        tp.close()


def render_page(data: bytes, page_no: int, highlights: list[str] | None = None, scale: float = 1.5
                ) -> tuple[bytes, int]:
    """Render 1-based `page_no` to PNG, painting highlight boxes. Returns (png, n_highlights)."""
    scale = max(0.5, min(scale, MAX_SCALE))
    pdf = pdfium.PdfDocument(data)
    try:
        if not 1 <= page_no <= len(pdf):
            raise IndexError(f"page {page_no} out of range 1..{len(pdf)}")
        page = pdf[page_no - 1]
        _, height = page.get_size()
        img = page.render(scale=scale).to_pil().convert("RGBA")
        rects = find_rects(page, highlights or [])
        if rects:
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            pad = 1.5 * scale
            for l, b, r, t in rects:  # PDF origin is bottom-left; image origin is top-left
                draw.rectangle([l * scale - pad, (height - t) * scale - pad, r * scale + pad, (height - b) * scale + pad],
                               fill=HIGHLIGHT, outline=(230, 150, 0, 255), width=max(1, int(scale)))
            img = Image.alpha_composite(img, overlay)
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="PNG", optimize=True)
        return buf.getvalue(), len(rects)
    finally:
        pdf.close()
