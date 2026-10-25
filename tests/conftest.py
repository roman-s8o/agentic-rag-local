from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image
from qdrant_client import QdrantClient

from agentic_rag.config import IngestionSettings, RetrievalSettings
from agentic_rag.ingestion.docling_parser import ParsedDocument
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever, QdrantStore
from tests.fakes import FakeImageEmbedder, FakeTextEncoder


class NoDoclingParser:
    def parse(self, path: Path) -> ParsedDocument:
        raise AssertionError(f"Docling should not be used for {path}")


@pytest.fixture
def store() -> QdrantStore:
    return QdrantStore(QdrantClient(":memory:"), "test")


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "inverter.txt").write_text(
        "The Helios H5 inverter has a peak efficiency of 98.2 percent.\n\n"
        "Maximum input voltage is 600 volts and it supports two MPPT trackers."
    )
    (root / "policy.txt").write_text(
        "Employees may book economy class for flights shorter than six hours.\n\n"
        "Hotel stays are capped at 180 euros per night."
    )
    Image.new("RGB", (200, 120), "white").save(root / "chart.png")
    return root


@pytest.fixture
def pipeline(store: QdrantStore, tmp_path: Path) -> IngestionPipeline:
    return IngestionPipeline(
        store=store,
        text_encoder=FakeTextEncoder(),
        settings=IngestionSettings(images_dir=tmp_path / "images", caption_images=False),
        parser=NoDoclingParser(),
        image_embedder=FakeImageEmbedder(),
    )


@pytest.fixture
def retriever(store: QdrantStore) -> HybridRetriever:
    return HybridRetriever(
        store,
        FakeTextEncoder(),
        RetrievalSettings(prefetch_limit=10, limit=5),
        image_embedder=FakeImageEmbedder(),
    )
