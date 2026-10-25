from __future__ import annotations

import base64
import binascii
import json
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from agentic_rag import __version__
from agentic_rag.config import get_settings
from agentic_rag.ingestion import SUPPORTED_EXTENSIONS
from agentic_rag.schema import Citation, content_hash
from agentic_rag.services import Services


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = None
    image_base64: str | None = Field(default=None, description="PNG/JPEG, raw or data URI")
    sources: list[str] | None = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    grounded: bool | None
    route: str | None
    queries: list[str]
    thread_id: str
    trace_id: str | None


class IngestResponse(BaseModel):
    files: int
    text_chunks: int
    image_chunks: int
    failed: dict[str, str]


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _save_query_image(payload: str, directory: Path) -> str:
    data = payload.split(",", 1)[1] if payload.startswith("data:") else payload
    try:
        raw = base64.b64decode(data, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(422, "image_base64 is not valid base64") from exc
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"query-{content_hash(raw)[:16]}.png"
    path.write_bytes(raw)
    return str(path)


def _sse(events: Iterator[dict[str, Any]]) -> Iterator[str]:
    for event in events:
        yield f"event: {event['type']}\ndata: {json.dumps(event, default=str)}\n\n"


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.services = services or Services(get_settings())
        yield

    app = FastAPI(title="Agentic RAG", version=__version__, lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "version": __version__}

    @app.get("/stats")
    def stats(svc: ServicesDep) -> dict[str, Any]:
        return {"collection": svc.settings.qdrant.collection, "points": svc.store.count()}

    @app.post("/ingest", response_model=IngestResponse)
    def ingest(svc: ServicesDep, files: Annotated[list[UploadFile], File()]) -> IngestResponse:
        uploads = svc.settings.ingestion.uploads_dir
        uploads.mkdir(parents=True, exist_ok=True)
        svc.pipeline.ensure_collection()
        result = IngestResponse(files=0, text_chunks=0, image_chunks=0, failed={})
        for upload in files:
            name = Path(upload.filename or "upload").name
            if Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                result.failed[name] = "unsupported file type"
                continue
            path = uploads / name
            path.write_bytes(upload.file.read())
            try:
                texts, images = svc.pipeline.ingest_file(path, name)
            except Exception as exc:
                result.failed[name] = str(exc)
                continue
            result.files += 1
            result.text_chunks += texts
            result.image_chunks += images
        return result

    def _image(svc: Services, req: QueryRequest) -> str | None:
        if not req.image_base64:
            return None
        return _save_query_image(req.image_base64, svc.settings.ingestion.uploads_dir / "queries")

    @app.post("/query", response_model=QueryResponse)
    def query(svc: ServicesDep, req: QueryRequest) -> QueryResponse:
        r = svc.ask(req.question, _image(svc, req), req.thread_id, req.sources)
        return QueryResponse(
            answer=r.answer,
            citations=r.citations,
            grounded=r.grounded,
            route=r.route,
            queries=r.queries,
            thread_id=r.thread_id,
            trace_id=r.trace_id,
        )

    @app.post("/query/stream")
    def query_stream(svc: ServicesDep, req: QueryRequest) -> StreamingResponse:
        events = svc.stream(req.question, _image(svc, req), req.thread_id, req.sources)
        return StreamingResponse(
            _sse(events),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
