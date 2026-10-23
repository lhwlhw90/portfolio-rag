import pytest

from rag.ingest import UnsupportedFileType, chunk_pages, parse_document, split_text


def test_split_respects_size_and_covers_text():
    text = " ".join(f"word{i}." for i in range(400))
    parts = split_text(text, chunk_size=200, overlap=40)
    assert len(parts) > 5
    assert all(len(p) <= 200 for _, p in parts)
    assert parts[0][1].startswith("word0.") and parts[-1][1].endswith("word399.")


def test_split_overlap_shares_content():
    text = " ".join(f"tok{i}" for i in range(200))
    parts = [p for _, p in split_text(text, chunk_size=100, overlap=30)]
    for a, b in zip(parts, parts[1:]):
        assert a.split()[-1] in b.split()  # last word of a chunk reappears in the next one


def test_split_prefers_paragraph_boundary():
    text = "A" * 70 + "\n\n" + "B" * 70
    parts = [p for _, p in split_text(text, chunk_size=100, overlap=0)]
    assert parts == ["A" * 70, "B" * 70]


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_split_rejects_bad_params(size, overlap):
    with pytest.raises(ValueError):
        split_text("hello", size, overlap)


def test_empty_text_yields_no_chunks():
    assert split_text("   \n\n ", 100, 10) == []


def test_parse_txt_and_md():
    assert parse_document("a.txt", b"hello")[0].text == "hello"
    pages = parse_document("a.MD", "# Título\nok".encode())
    assert pages[0].page is None and "Título" in pages[0].text


def test_parse_pdf_keeps_page_numbers(pdf_bytes):
    pages = parse_document("policy.pdf", pdf_bytes)
    assert [p.page for p in pages] == [1, 2]
    assert "14 characters" in pages[0].text and "1 hour" in pages[1].text


def test_unsupported_type():
    with pytest.raises(UnsupportedFileType):
        parse_document("x.docx", b"...")


def test_chunk_metadata(pdf_bytes):
    chunks = chunk_pages("d1", "policy.pdf", parse_document("policy.pdf", pdf_bytes), 120, 20)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.doc_name == "policy.pdf" and c.chunk_id.startswith("d1:") for c in chunks)
