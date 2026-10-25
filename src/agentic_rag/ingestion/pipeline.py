from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from PIL import Image as PILImage
from PIL.Image import Image

from agentic_rag.config import IngestionSettings
from agentic_rag.embeddings import ImageEmbedder, TextEncoder
from agentic_rag.ingestion.chunking import TextPiece, split_text
from agentic_rag.ingestion.docling_parser import (
    DOCLING_EXTENSIONS,
    ParsedDocument,
    ParsedPicture,
)
from agentic_rag.ingestion.images import (
    IMAGE_EXTENSIONS,
    ImageCaptioner,
    is_informative,
    save_image,
)
from agentic_rag.retrieval.qdrant_store import QdrantStore, VectorRecord
from agentic_rag.schema import Chunk, chunk_id, content_hash

logger = logging.getLogger(__name__)

TEXT_EXTENSIONS = frozenset({".txt"})
SUPPORTED_EXTENSIONS = DOCLING_EXTENSIONS | IMAGE_EXTENSIONS | TEXT_EXTENSIONS


class DocumentParser(Protocol):
    def parse(self, path: Path) -> ParsedDocument: ...


@dataclass
class IngestReport:
    files: int = 0
    text_chunks: int = 0
    image_chunks: int = 0
    failed: dict[str, str] = field(default_factory=dict)


@dataclass
class _PendingImage:
    chunk: Chunk
    image: Image


class IngestionPipeline:
    def __init__(
        self,
        store: QdrantStore,
        text_encoder: TextEncoder,
        settings: IngestionSettings,
        parser: DocumentParser,
        image_embedder: ImageEmbedder | None = None,
        captioner: ImageCaptioner | None = None,
    ) -> None:
        self.store = store
        self.text_encoder = text_encoder
        self.settings = settings
        self.parser = parser
        self.image_embedder = image_embedder
        self.captioner = captioner

    def ensure_collection(self, recreate: bool = False) -> None:
        self.store.ensure_collection(
            dense_dim=self.text_encoder.dim,
            sparse_idf=self.text_encoder.sparse_requires_idf,
            image_dim=self.image_embedder.dim if self.image_embedder else None,
            recreate=recreate,
        )

    @staticmethod
    def discover(path: Path) -> list[Path]:
        if path.is_file():
            return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []
        return sorted(
            p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    def ingest(self, path: Path, recreate: bool = False) -> IngestReport:
        self.ensure_collection(recreate=recreate)
        root = path if path.is_dir() else path.parent
        report = IngestReport()
        for file in self.discover(path):
            source = file.relative_to(root).as_posix()
            try:
                texts, images = self.ingest_file(file, source)
            except Exception as exc:
                logger.exception("Failed to ingest %s", file)
                report.failed[source] = str(exc)
                continue
            report.files += 1
            report.text_chunks += texts
            report.image_chunks += images
        return report

    def ingest_file(self, file: Path, source: str) -> tuple[int, int]:
        texts, images = self._build_chunks(file, source)
        records = self._encode(texts, images)
        self.store.delete_source(source)
        self.store.upsert(records)
        logger.info("%s: %d text, %d image chunks", source, len(texts), len(images))
        return len(texts), len(images)

    def _build_chunks(self, file: Path, source: str) -> tuple[list[Chunk], list[_PendingImage]]:
        suffix = file.suffix.lower()
        if suffix in IMAGE_EXTENSIONS:
            with PILImage.open(file) as img:
                pictures = [ParsedPicture(image=img.copy())]
            parsed = ParsedDocument(pictures=pictures)
        elif suffix in TEXT_EXTENSIONS:
            parsed = ParsedDocument(texts=[TextPiece(text=t) for t in split_text(file.read_text())])
        else:
            parsed = self.parser.parse(file)

        texts = [
            Chunk(
                id=chunk_id(source, "text", i, content_hash(piece.text)),
                text=piece.text,
                source=source,
                page=piece.page,
                headings=piece.headings,
            )
            for i, piece in enumerate(parsed.texts)
        ]
        context = " / ".join(dict.fromkeys(h for t in texts for h in t.headings))[:500]
        images = list(self._image_chunks(parsed.pictures, source, context))
        return texts, images

    def _image_chunks(
        self, pictures: Iterable[ParsedPicture], source: str, context: str
    ) -> Iterable[_PendingImage]:
        for i, pic in enumerate(pictures):
            if not is_informative(pic.image, self.settings.min_image_px):
                continue
            path, digest = save_image(pic.image, self.settings.images_dir)
            description = pic.caption
            if self.captioner and self.settings.caption_images:
                generated = self.captioner.caption(path, context)
                description = f"{pic.caption}\n{generated}".strip()
            text = f"[Image] {description or f'Image from {source}'}"
            chunk = Chunk(
                id=chunk_id(source, "image", i, digest),
                text=text,
                source=source,
                modality="image",
                page=pic.page,
                image_path=str(path),
            )
            yield _PendingImage(chunk=chunk, image=pic.image)

    def _encode(self, texts: list[Chunk], images: list[_PendingImage]) -> list[VectorRecord]:
        chunks = [*texts, *(p.chunk for p in images)]
        if not chunks:
            return []
        encodings = self.text_encoder.encode_documents([c.text for c in chunks])
        image_vectors: list[list[list[float]] | None] = [None] * len(texts)
        if self.image_embedder and images:
            image_vectors.extend(self.image_embedder.embed_images([p.image for p in images]))
        else:
            image_vectors.extend([None] * len(images))
        return [
            VectorRecord(chunk=c, dense=e.dense, sparse=e.sparse, image=v)
            for c, e, v in zip(chunks, encodings, image_vectors, strict=True)
        ]
