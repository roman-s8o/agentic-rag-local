from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class SparseEmbedding:
    indices: list[int] = field(default_factory=list)
    values: list[float] = field(default_factory=list)

    @classmethod
    def from_weights(cls, weights: Mapping[str | int, float]) -> SparseEmbedding:
        merged: dict[int, float] = {}
        for key, value in weights.items():
            idx = int(key)
            merged[idx] = max(merged.get(idx, 0.0), float(value))
        items = sorted(merged.items())
        return cls(indices=[i for i, _ in items], values=[v for _, v in items])


class SparseEmbedder(Protocol):
    requires_idf: bool

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]: ...

    def embed_query(self, text: str) -> SparseEmbedding: ...


class BM25SparseEmbedder:
    """BM25 term weights via FastEmbed; Qdrant applies IDF server-side."""

    requires_idf = True

    def __init__(self, model: str = "Qdrant/bm25") -> None:
        from fastembed import SparseTextEmbedding

        self._model = SparseTextEmbedding(model_name=model)

    def embed_documents(self, texts: list[str]) -> list[SparseEmbedding]:
        return [
            SparseEmbedding(e.indices.tolist(), e.values.tolist()) for e in self._model.embed(texts)
        ]

    def embed_query(self, text: str) -> SparseEmbedding:
        e = next(iter(self._model.query_embed(text)))
        return SparseEmbedding(e.indices.tolist(), e.values.tolist())
