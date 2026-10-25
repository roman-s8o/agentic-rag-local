from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from agentic_rag.config import RerankerSettings
from agentic_rag.embeddings.device import resolve_device
from agentic_rag.schema import ScoredChunk


class Reranker(Protocol):
    def rerank(self, query: str, docs: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]: ...


def _apply_scores(
    docs: Sequence[ScoredChunk], scores: Sequence[float], top_n: int
) -> list[ScoredChunk]:
    rescored = [
        d.model_copy(update={"rerank_score": float(s)}) for d, s in zip(docs, scores, strict=True)
    ]
    rescored.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
    return rescored[:top_n]


class NoopReranker:
    def rerank(self, query: str, docs: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        return list(docs[:top_n])


class CrossEncoderReranker:
    """Local cross-encoder (bge-reranker-v2-m3, mxbai-rerank, ...). Images are scored by caption."""

    def __init__(self, model: str, device: str | None = None, max_length: int = 1024) -> None:
        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(model, device=resolve_device(device), max_length=max_length)

    def rerank(self, query: str, docs: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not docs:
            return []
        scores = self._model.predict([(query, d.chunk.text) for d in docs])
        return _apply_scores(docs, [float(s) for s in scores], top_n)


class HttpReranker:
    """Cohere/Jina-compatible /rerank endpoint (vLLM, Infinity, TEI)."""

    def __init__(self, base_url: str, model: str, timeout_s: float = 60.0) -> None:
        self.url = base_url.rstrip("/") + "/rerank"
        self.model = model
        self._client = httpx.Client(timeout=timeout_s)

    def rerank(self, query: str, docs: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        if not docs:
            return []
        response = self._client.post(
            self.url,
            json={"model": self.model, "query": query, "documents": [d.chunk.text for d in docs]},
        )
        response.raise_for_status()
        scores = [0.0] * len(docs)
        for result in response.json()["results"]:
            scores[result["index"]] = result["relevance_score"]
        return _apply_scores(docs, scores, top_n)


def build_reranker(settings: RerankerSettings, device: str | None = None) -> Reranker:
    if settings.backend == "none":
        return NoopReranker()
    if settings.backend == "http":
        if not settings.base_url:
            raise ValueError("reranker.base_url is required for the http backend")
        return HttpReranker(settings.base_url, settings.model)
    return CrossEncoderReranker(settings.model, device)
