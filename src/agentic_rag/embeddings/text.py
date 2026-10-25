from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.embeddings.device import resolve_device
from agentic_rag.embeddings.sparse import SparseEmbedder, SparseEmbedding


@dataclass(frozen=True)
class TextEncoding:
    dense: list[float]
    sparse: SparseEmbedding


class TextEncoder(Protocol):
    dim: int
    sparse_requires_idf: bool

    def encode_documents(self, texts: list[str]) -> list[TextEncoding]: ...

    def encode_query(self, text: str) -> TextEncoding: ...


class BGEM3Encoder:
    """Dense and learned-sparse vectors from a single BGE-M3 forward pass."""

    sparse_requires_idf = False

    def __init__(
        self,
        model: str = "BAAI/bge-m3",
        device: str | None = None,
        batch_size: int = 16,
        max_length: int = 8192,
    ) -> None:
        from FlagEmbedding import BGEM3FlagModel

        dev = resolve_device(device)
        self._model = BGEM3FlagModel(model, use_fp16=dev == "cuda", devices=[dev])
        self.batch_size = batch_size
        self.max_length = max_length
        self.dim = 1024

    def _encode(self, texts: list[str], max_length: int) -> list[TextEncoding]:
        out: Any = self._model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return [
            TextEncoding(dense=[float(x) for x in dense], sparse=SparseEmbedding.from_weights(w))
            for dense, w in zip(out["dense_vecs"], out["lexical_weights"], strict=True)
        ]

    def encode_documents(self, texts: list[str]) -> list[TextEncoding]:
        return self._encode(texts, self.max_length) if texts else []

    def encode_query(self, text: str) -> TextEncoding:
        return self._encode([text], 512)[0]


class CompositeTextEncoder:
    """Any sentence-transformers dense model combined with a separate sparse model."""

    def __init__(
        self,
        dense_model: str,
        sparse: SparseEmbedder,
        device: str | None = None,
        batch_size: int = 16,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self._dense = SentenceTransformer(dense_model, device=resolve_device(device))
        self._sparse = sparse
        self.batch_size = batch_size
        self.dim = int(self._dense.get_sentence_embedding_dimension() or 0)
        self.sparse_requires_idf = sparse.requires_idf

    def encode_documents(self, texts: list[str]) -> list[TextEncoding]:
        if not texts:
            return []
        dense = self._dense.encode(texts, batch_size=self.batch_size, normalize_embeddings=True)
        sparse = self._sparse.embed_documents(texts)
        return [
            TextEncoding(dense=d.tolist(), sparse=s) for d, s in zip(dense, sparse, strict=True)
        ]

    def encode_query(self, text: str) -> TextEncoding:
        dense = self._dense.encode([text], normalize_embeddings=True)[0]
        return TextEncoding(dense=dense.tolist(), sparse=self._sparse.embed_query(text))
