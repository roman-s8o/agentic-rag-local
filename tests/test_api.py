import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agentic_rag.api import create_app
from agentic_rag.config import IngestionSettings, Settings
from agentic_rag.schema import Citation
from agentic_rag.services import AgentResult


class FakePipeline:
    def __init__(self) -> None:
        self.ingested: list[str] = []

    def ensure_collection(self) -> None:
        pass

    def ingest_file(self, path: Path, source: str) -> tuple[int, int]:
        self.ingested.append(source)
        return 2, 0


class FakeServices:
    def __init__(self, tmp_path: Path) -> None:
        self.settings = Settings(ingestion=IngestionSettings(uploads_dir=tmp_path / "uploads"))
        self.pipeline = FakePipeline()
        self.images: list[str | None] = []

    def ask(self, question: str, image: str | None = None, *_: Any) -> AgentResult:
        self.images.append(image)
        return AgentResult(
            answer="98.2 % [1]",
            citations=[Citation(index=1, source="manual.md")],
            grounded=True,
            route="retrieve",
            queries=[question],
            thread_id="t1",
        )

    def stream(self, question: str, *_: Any) -> Iterator[dict[str, Any]]:
        yield {"type": "node", "node": "analyze"}
        yield {"type": "token", "text": "98.2"}
        yield {"type": "final", "answer": "98.2", "citations": [], "thread_id": "t1"}


@pytest.fixture
def services(tmp_path: Path) -> FakeServices:
    return FakeServices(tmp_path)


@pytest.fixture
def client(services: FakeServices) -> Iterator[TestClient]:
    with TestClient(create_app(services)) as c:  # type: ignore[arg-type]
        yield c


def test_health(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"


def test_query_with_image(client: TestClient, services: FakeServices) -> None:
    img = base64.b64encode(b"\x89PNG fake").decode()
    response = client.post(
        "/query", json={"question": "efficiency?", "image_base64": f"data:image/png;base64,{img}"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["citations"][0]["source"] == "manual.md"
    assert services.images[0] is not None
    assert Path(services.images[0]).read_bytes() == b"\x89PNG fake"


def test_query_rejects_bad_image(client: TestClient) -> None:
    response = client.post("/query", json={"question": "q", "image_base64": "%%%"})
    assert response.status_code == 422


def test_stream_emits_sse(client: TestClient) -> None:
    with client.stream("POST", "/query/stream", json={"question": "q"}) as response:
        body = "".join(response.iter_text())
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    assert [e["type"] for e in events] == ["node", "token", "final"]


def test_ingest_upload(client: TestClient, services: FakeServices) -> None:
    response = client.post(
        "/ingest",
        files=[
            ("files", ("../../evil.txt", b"hello", "text/plain")),
            ("files", ("binary.exe", b"x", "application/octet-stream")),
        ],
    )
    body = response.json()
    assert body["files"] == 1
    assert body["text_chunks"] == 2
    assert "binary.exe" in body["failed"]
    assert services.pipeline.ingested == ["evil.txt"]
    assert (services.settings.ingestion.uploads_dir / "evil.txt").exists()
