from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from agentic_rag.config import IngestionSettings, RetrievalSettings
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever, QdrantStore
from tests.conftest import NoDoclingParser
from tests.fakes import FakeImageEmbedder, FakeTextEncoder

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def server_client() -> Iterator[QdrantClient]:
    from testcontainers.community.qdrant import QdrantContainer

    with QdrantContainer("qdrant/qdrant:v1.20.0") as container:
        yield container.get_client()


def test_hybrid_search_against_server(
    server_client: QdrantClient, corpus: Path, tmp_path: Path
) -> None:
    store = QdrantStore(server_client, "integration")
    pipeline = IngestionPipeline(
        store=store,
        text_encoder=FakeTextEncoder(),
        settings=IngestionSettings(images_dir=tmp_path / "images", caption_images=False),
        parser=NoDoclingParser(),
        image_embedder=FakeImageEmbedder(),
    )
    report = pipeline.ingest(corpus, recreate=True)
    assert report.files == 3
    assert store.count() == 3

    retriever = HybridRetriever(
        store, FakeTextEncoder(), RetrievalSettings(limit=5), FakeImageEmbedder()
    )
    results = retriever.retrieve("maximum input voltage MPPT")
    assert results[0].chunk.source == "inverter.txt"

    filtered = retriever.retrieve("voltage", sources=["policy.txt"])
    assert {r.chunk.source for r in filtered} == {"policy.txt"}
