"""FastAPI app: upload, list/delete documents, ask, plus a static web demo."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .ingest import UnsupportedFileType
from .service import EmptyDocument, RAGService

STATIC = Path(__file__).parent / "static"


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int | None = Field(None, ge=1, le=20)


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
                "answerer": s.answerer.name, "documents": len(s.store.list_documents()),
                "chunks": len(s.store.chunks)}

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
        if not svc().store.delete_document(doc_id):
            raise HTTPException(404, "document not found")

    @app.post("/ask", response_model=AskResponse)
    def ask(req: AskRequest):
        return svc().ask(req.question, req.top_k)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def app_factory() -> FastAPI:  # used by `uvicorn rag.api:app_factory --factory`
    return create_app()
