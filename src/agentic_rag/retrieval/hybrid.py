from __future__ import annotations

from collections.abc import Sequence

from agentic_rag.config import RetrievalSettings
from agentic_rag.embeddings import ImageEmbedder, TextEncoder
from agentic_rag.retrieval.qdrant_store import QdrantStore
from agentic_rag.schema import ScoredChunk


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[ScoredChunk]], k: int = 60, limit: int | None = None
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            key = item.chunk.id
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, item)
    fused = sorted(scores, key=lambda key: scores[key], reverse=True)
    out = [best[key].model_copy(update={"score": scores[key]}) for key in fused]
    return out[:limit] if limit else out


class HybridRetriever:
    """Dense + sparse (+ late-interaction image) search fused with RRF inside Qdrant;
    multiple query variants are fused again client-side."""

    def __init__(
        self,
        store: QdrantStore,
        text_encoder: TextEncoder,
        settings: RetrievalSettings,
        image_embedder: ImageEmbedder | None = None,
    ) -> None:
        self.store = store
        self.text_encoder = text_encoder
        self.image_embedder = image_embedder
        self.settings = settings

    def retrieve(self, query: str, sources: Sequence[str] | None = None) -> list[ScoredChunk]:
        enc = self.text_encoder.encode_query(query)
        image_query = self.image_embedder.embed_query(query) if self.image_embedder else None
        return self.store.hybrid_search(
            dense=enc.dense,
            sparse=enc.sparse,
            image=image_query,
            prefetch_limit=self.settings.prefetch_limit,
            limit=self.settings.limit,
            sources=sources,
        )

    def retrieve_many(
        self, queries: Sequence[str], sources: Sequence[str] | None = None
    ) -> list[ScoredChunk]:
        unique = list(dict.fromkeys(q.strip() for q in queries if q.strip()))
        if len(unique) == 1:
            return self.retrieve(unique[0], sources)
        rankings = [self.retrieve(q, sources) for q in unique]
        return reciprocal_rank_fusion(rankings, k=self.settings.rrf_k, limit=self.settings.limit)
