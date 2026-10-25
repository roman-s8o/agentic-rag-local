from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agentic_rag.models.hf_downloader import (
    HFModelDownloader,
    ModelAccessError,
    ModelRegistry,
)


class FakeApi:
    def __init__(self, gated: bool = False, license_: str | None = "apache-2.0") -> None:
        self.gated = gated
        self.license = license_

    def model_info(self, repo_id: str, revision: str | None = None) -> Any:
        return SimpleNamespace(
            sha="abc123",
            gated="manual" if self.gated else False,
            card_data={"license": self.license},
            pipeline_tag="feature-extraction",
        )


def test_pull_passes_patterns_and_cache_dir(tmp_path: Path) -> None:
    calls: list[dict[str, Any]] = []

    def fake_download(**kwargs: Any) -> str:
        calls.append(kwargs)
        return str(tmp_path / "snap")

    downloader = HFModelDownloader(tmp_path, api=FakeApi(), download_fn=fake_download)  # type: ignore[arg-type]
    result = downloader.pull("BAAI/bge-m3", revision="main", include=["*.json"], exclude=["*.onnx"])

    assert result.path == tmp_path / "snap"
    assert result.info is not None
    assert result.info.license == "apache-2.0"
    assert calls[0]["cache_dir"] == tmp_path
    assert calls[0]["allow_patterns"] == ["*.json"]
    assert calls[0]["ignore_patterns"] == ["*.onnx"]
    assert calls[0]["revision"] == "main"


def test_gated_model_without_token_fails(tmp_path: Path) -> None:
    downloader = HFModelDownloader(
        tmp_path,
        token=None,
        api=FakeApi(gated=True),  # type: ignore[arg-type]
        download_fn=lambda **_: "unused",
    )
    with pytest.raises(ModelAccessError, match="gated"):
        downloader.pull("meta-llama/Llama-3.1-8B")


def test_registry_lookup(tmp_path: Path) -> None:
    path = tmp_path / "models.yaml"
    path.write_text(
        "models:\n"
        "  - {role: embedding, kind: embedding, repo_id: BAAI/bge-m3}\n"
        "  - {role: generator, kind: vlm, repo_id: Qwen/Qwen3-VL-8B-Instruct, enabled: false}\n"
    )
    registry = ModelRegistry.load(path)
    embedding = registry.get("embedding")
    assert embedding is not None
    assert embedding.repo_id == "BAAI/bge-m3"
    assert registry.get("Qwen/Qwen3-VL-8B-Instruct") is not None
    assert [m.role for m in registry.enabled()] == ["embedding"]


def test_project_registry_is_valid() -> None:
    registry = ModelRegistry.load(Path("configs/models.yaml"))
    assert registry.get("embedding") is not None
    assert registry.get("reranker") is not None
