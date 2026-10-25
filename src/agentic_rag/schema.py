from __future__ import annotations

import hashlib
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

Modality = Literal["text", "image"]


def content_hash(data: str | bytes) -> str:
    raw = data.encode() if isinstance(data, str) else data
    return hashlib.sha256(raw).hexdigest()


def chunk_id(source: str, modality: Modality, index: int, digest: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source}|{modality}|{index}|{digest}"))


class Chunk(BaseModel):
    id: str
    text: str
    source: str
    modality: Modality = "text"
    page: int | None = None
    headings: list[str] = Field(default_factory=list)
    image_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def label(self) -> str:
        loc = f" p.{self.page}" if self.page else ""
        return f"{self.source}{loc}"


class ScoredChunk(BaseModel):
    chunk: Chunk
    score: float
    rerank_score: float | None = None


class Citation(BaseModel):
    index: int
    source: str
    page: int | None = None
    modality: Modality = "text"
    image_path: str | None = None
    snippet: str = ""
