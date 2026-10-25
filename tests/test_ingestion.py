from pathlib import Path

import pytest

from agentic_rag.ingestion.chunking import split_text
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.retrieval import QdrantStore


def test_split_text_respects_limits() -> None:
    text = "\n\n".join(f"Paragraph {i} " + "word " * 50 for i in range(10))
    chunks = split_text(text, max_chars=400, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    assert "Paragraph 0" in chunks[0]
    assert "Paragraph 9" in chunks[-1]


def test_split_text_long_paragraph() -> None:
    chunks = split_text("x" * 1000, max_chars=300, overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert sum(len(c) for c in chunks) >= 1000


def test_split_text_rejects_bad_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        split_text("a", max_chars=10, overlap=10)


def test_ingest_text_and_images(
    pipeline: IngestionPipeline, store: QdrantStore, corpus: Path
) -> None:
    report = pipeline.ingest(corpus)
    assert report.files == 3
    assert report.failed == {}
    assert report.text_chunks == 2
    assert report.image_chunks == 1
    assert store.count() == 3
    assert store.has_image_vectors()


def test_reingest_is_idempotent(
    pipeline: IngestionPipeline, store: QdrantStore, corpus: Path
) -> None:
    pipeline.ingest(corpus)
    pipeline.ingest(corpus)
    assert store.count() == 3


def test_tiny_images_are_skipped(
    pipeline: IngestionPipeline, store: QdrantStore, tmp_path: Path
) -> None:
    from PIL import Image

    icon = tmp_path / "icon.png"
    Image.new("RGB", (16, 16)).save(icon)
    report = pipeline.ingest(icon)
    assert report.image_chunks == 0
