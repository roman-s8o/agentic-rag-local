from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from agentic_rag.config import QdrantSettings
from agentic_rag.embeddings.image import MultiVector
from agentic_rag.embeddings.sparse import SparseEmbedding
from agentic_rag.schema import Chunk, ScoredChunk

logger = logging.getLogger(__name__)

DENSE = "dense"
SPARSE = "sparse"
IMAGE = "image"


@dataclass(frozen=True)
class VectorRecord:
    chunk: Chunk
    dense: list[float]
    sparse: SparseEmbedding
    image: MultiVector | None = None


def source_filter(sources: Sequence[str] | None) -> qm.Filter | None:
    if not sources:
        return None
    return qm.Filter(must=[qm.FieldCondition(key="source", match=qm.MatchAny(any=list(sources)))])


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self.client = client
        self.collection = collection

    @classmethod
    def from_settings(cls, settings: QdrantSettings) -> QdrantStore:
        if settings.path is not None:
            client = QdrantClient(path=str(settings.path))
        else:
            api_key = settings.api_key.get_secret_value() if settings.api_key else None
            client = QdrantClient(url=settings.url, api_key=api_key)
        return cls(client, settings.collection)

    def exists(self) -> bool:
        return self.client.collection_exists(self.collection)

    def has_image_vectors(self) -> bool:
        if not self.exists():
            return False
        vectors = self.client.get_collection(self.collection).config.params.vectors
        return isinstance(vectors, dict) and IMAGE in vectors

    def ensure_collection(
        self,
        dense_dim: int,
        sparse_idf: bool,
        image_dim: int | None = None,
        recreate: bool = False,
    ) -> None:
        if recreate and self.exists():
            self.client.delete_collection(self.collection)
        if self.exists():
            return
        vectors: dict[str, qm.VectorParams] = {
            DENSE: qm.VectorParams(size=dense_dim, distance=qm.Distance.COSINE)
        }
        if image_dim:
            vectors[IMAGE] = qm.VectorParams(
                size=image_dim,
                distance=qm.Distance.COSINE,
                multivector_config=qm.MultiVectorConfig(
                    comparator=qm.MultiVectorComparator.MAX_SIM
                ),
            )
        self.client.create_collection(
            self.collection,
            vectors_config=vectors,
            sparse_vectors_config={
                SPARSE: qm.SparseVectorParams(modifier=qm.Modifier.IDF if sparse_idf else None)
            },
        )
        for field in ("source", "modality"):
            self.client.create_payload_index(
                self.collection, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
            )
        logger.info("Created collection %s", self.collection)

    def upsert(self, records: Sequence[VectorRecord], batch_size: int = 64) -> None:
        for i in range(0, len(records), batch_size):
            points = []
            for r in records[i : i + batch_size]:
                vector: dict[str, qm.Vector] = {
                    DENSE: r.dense,
                    SPARSE: qm.SparseVector(indices=r.sparse.indices, values=r.sparse.values),
                }
                if r.image is not None:
                    vector[IMAGE] = r.image
                points.append(
                    qm.PointStruct(
                        id=r.chunk.id, vector=vector, payload=r.chunk.model_dump(mode="json")
                    )
                )
            self.client.upsert(self.collection, points=points, wait=True)

    def delete_source(self, source: str) -> None:
        if not self.exists():
            return
        flt = source_filter([source])
        assert flt is not None
        self.client.delete(self.collection, points_selector=qm.FilterSelector(filter=flt))

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count if self.exists() else 0

    def hybrid_search(
        self,
        dense: list[float],
        sparse: SparseEmbedding,
        image: MultiVector | None = None,
        prefetch_limit: int = 30,
        limit: int = 20,
        sources: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        flt = source_filter(sources)
        prefetch = [
            qm.Prefetch(query=dense, using=DENSE, limit=prefetch_limit, filter=flt),
            qm.Prefetch(
                query=qm.SparseVector(indices=sparse.indices, values=sparse.values),
                using=SPARSE,
                limit=prefetch_limit,
                filter=flt,
            ),
        ]
        if image is not None and self.has_image_vectors():
            prefetch.append(qm.Prefetch(query=image, using=IMAGE, limit=prefetch_limit, filter=flt))
        response = self.client.query_points(
            self.collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return [
            ScoredChunk(chunk=Chunk.model_validate(p.payload), score=p.score)
            for p in response.points
            if p.payload
        ]
