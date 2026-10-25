from agentic_rag.config import EmbeddingSettings
from agentic_rag.embeddings.image import ColQwenEmbedder, ImageEmbedder, MultiVector
from agentic_rag.embeddings.sparse import BM25SparseEmbedder, SparseEmbedder, SparseEmbedding
from agentic_rag.embeddings.text import (
    BGEM3Encoder,
    CompositeTextEncoder,
    TextEncoder,
    TextEncoding,
)


def build_text_encoder(settings: EmbeddingSettings) -> TextEncoder:
    if settings.text_backend == "bgem3":
        return BGEM3Encoder(settings.dense_model, settings.device, settings.batch_size)
    if settings.sparse_backend == "bgem3":
        raise ValueError("sparse_backend=bgem3 requires text_backend=bgem3; use bm25 instead.")
    return CompositeTextEncoder(
        settings.dense_model,
        BM25SparseEmbedder(settings.sparse_model),
        settings.device,
        settings.batch_size,
    )


def build_image_embedder(settings: EmbeddingSettings) -> ImageEmbedder | None:
    if not settings.image_enabled:
        return None
    return ColQwenEmbedder(settings.image_model, settings.device)


__all__ = [
    "BGEM3Encoder",
    "BM25SparseEmbedder",
    "ColQwenEmbedder",
    "CompositeTextEncoder",
    "ImageEmbedder",
    "MultiVector",
    "SparseEmbedder",
    "SparseEmbedding",
    "TextEncoder",
    "TextEncoding",
    "build_image_embedder",
    "build_text_encoder",
]
