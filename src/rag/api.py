"""FastAPI app: upload, list/delete documents, ask, plus a static web demo."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .ingest import UnsupportedFileType
from .service import EmptyDocument, RAGService

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)
    mode: Literal["hybrid", "vector", "bm25"] = "hybrid"
    answerer: Literal["default", "extractive", "llm"] = "default"


class CitationOut(BaseModel):
    index: int
    doc_id: str
    doc_name: str
    chunk_id: str
    page: int | None
    snippet: str
    score: float


class AskResponse(BaseModel):
    question: str
    answer: str
    backend: str
    refused: bool = False
    refusal_reason: str | None = None
    refusal_detail: str | None = None
    confidence: float | None = None
    min_score: float | None = None
    citation_mode: str = "explicit"
    mode: str = "hybrid"
    latency_ms: float | None = None
    citations: list[CitationOut]


def create_app(settings: Settings | None = None, service: RAGService | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="portfolio-rag", version=__version__)
    app.state.service = service  # lazily built so the model loads on first use / startup

    def svc() -> RAGService:
        if app.state.service is None:
            app.state.service = RAGService(settings)
        return app.state.service

    @app.get("/health")
    def health():
        s = svc()
        return {"status": "ok", "version": __version__, "embedder": s.embedder.name,
                "answerer": s.answerer.name, "llm_model": s.settings.llm_model,
                "llm_available": s.llm_available(), "min_score": s.settings.resolved_min_score,
                "documents": len(s.store.list_documents()), "chunks": len(s.store.chunks)}

    @app.post("/documents", status_code=201)
    async def upload(file: UploadFile = File(...)):
        data = await file.read()
        if len(data) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"File larger than {settings.max_upload_mb} MB")
        try:
            return svc().ingest(file.filename or "upload.txt", data)
        except UnsupportedFileType as e:
            raise HTTPException(415, str(e))
        except EmptyDocument as e:
            raise HTTPException(422, str(e))

    @app.get("/documents")
    def list_docs():
        return svc().store.list_documents()

    @app.delete("/documents/{doc_id}", status_code=204)
    def delete_doc(doc_id: str):
        if not svc().delete(doc_id):
            raise HTTPException(404, "document not found")

    def _doc_file(doc_id: str) -> tuple[dict, Path]:
        doc = svc().store.get_document(doc_id)
        if doc is None:
            raise HTTPException(404, "document not found")
        path = svc().file_path(doc_id)
        if path is None:
            raise HTTPException(404, "original file not stored (document ingested before files were kept)")
        return doc, path

    @app.get("/documents/{doc_id}/file")
    def doc_file(doc_id: str):
        doc, path = _doc_file(doc_id)
        media = "application/pdf" if path.suffix == ".pdf" else "text/plain; charset=utf-8"
        return FileResponse(path, media_type=media, filename=doc["name"],
                            content_disposition_type="inline")

    @app.get("/documents/{doc_id}/pages/{page}.png")
    def doc_page(doc_id: str, page: int, highlight: list[str] = Query(default=[]),
                 scale: float = Query(1.5, ge=0.5, le=3.0)):
        from .pdfview import render_page  # optional heavy import, only for PDF previews
        doc, path = _doc_file(doc_id)
        if path.suffix != ".pdf":
            raise HTTPException(400, "page rendering is only available for PDF documents")
        try:
            png, n = render_page(path.read_bytes(), page, highlight, scale)
        except IndexError as e:
            raise HTTPException(404, str(e))
        return Response(png, media_type="image/png",
                        headers={"X-Highlights": str(n), "Cache-Control": "private, max-age=300"})

    @app.get("/chunks/{chunk_id}")
    def chunk(chunk_id: str):
        c = svc().store.get_chunk(chunk_id)
        if c is None:
            raise HTTPException(404, "chunk not found")
        doc = svc().store.get_document(c.doc_id) or {}
        return {"chunk_id": c.chunk_id, "doc_id": c.doc_id, "doc_name": c.doc_name, "page": c.page,
                "pages": doc.get("pages"), "chunk_index": c.chunk_index, "text": c.text,
                "has_file": svc().file_path(c.doc_id) is not None}

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        return svc().ask(req.question, req.top_k, req.mode, req.answerer)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest):
        """Server-Sent Events: `retrieval`, `token`*, optional `reset`, then `done` (same body as /ask)."""
        events = svc().ask_stream(req.question, req.top_k, req.mode, req.answerer)

        def gen():
            for kind, payload in events:
                yield f"event: {kind}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def app_factory() -> FastAPI:  # used by `uvicorn rag.api:app_factory --factory`
    return create_app()
