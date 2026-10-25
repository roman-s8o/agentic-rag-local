from __future__ import annotations

import math
import re
import zlib
from collections import Counter
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage
from PIL.Image import Image
from pydantic import BaseModel

from agentic_rag.embeddings import SparseEmbedding, TextEncoding
from agentic_rag.providers.base import LLMProvider, T, message_text
from agentic_rag.schema import ScoredChunk

DIM = 64


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _bucket(token: str, size: int) -> int:
    return zlib.crc32(token.encode()) % size


class FakeTextEncoder:
    dim = DIM
    sparse_requires_idf = True

    def _encode(self, text: str) -> TextEncoding:
        counts = Counter(_tokens(text))
        dense = [0.0] * DIM
        for tok, n in counts.items():
            dense[_bucket(tok, DIM)] += n
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        sparse = SparseEmbedding.from_weights(
            {_bucket(t, 1_000_003): float(n) for t, n in counts.items()}
        )
        return TextEncoding(dense=[x / norm for x in dense], sparse=sparse)

    def encode_documents(self, texts: list[str]) -> list[TextEncoding]:
        return [self._encode(t) for t in texts]

    def encode_query(self, text: str) -> TextEncoding:
        return self._encode(text)


class FakeImageEmbedder:
    dim = 8

    def embed_images(self, images: list[Image]) -> list[list[list[float]]]:
        return [[[1.0] + [0.0] * 7, [0.0, 1.0] + [0.0] * 6] for _ in images]

    def embed_query(self, text: str) -> list[list[float]]:
        return [[1.0] + [0.0] * 7]


class KeywordReranker:
    def rerank(self, query: str, docs: Sequence[ScoredChunk], top_n: int) -> list[ScoredChunk]:
        q = set(_tokens(query))
        scored = [
            d.model_copy(update={"rerank_score": float(len(q & set(_tokens(d.chunk.text))))})
            for d in docs
        ]
        scored.sort(key=lambda d: d.rerank_score or 0.0, reverse=True)
        return scored[:top_n]


StructuredHandler = Callable[[Sequence[BaseMessage]], BaseModel]


class ScriptedLLM(LLMProvider):
    """Structured calls are answered by per-schema handlers; free text by a fixed list."""

    def __init__(
        self,
        structured_handlers: dict[str, StructuredHandler] | None = None,
        responses: list[str] | None = None,
    ) -> None:
        self.handlers = structured_handlers or {}
        self.responses = responses or ["answer"]
        self.calls: list[tuple[str, Any]] = []
        self._chat_model = FakeListChatModel(responses=self.responses)

    @property
    def chat_model(self) -> BaseChatModel:
        return self._chat_model

    def structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        self.calls.append((schema.__name__, messages))
        result = self.handlers[schema.__name__](messages)
        assert isinstance(result, schema)
        return result

    def complete(self, messages: Sequence[BaseMessage]) -> str:
        self.calls.append(("complete", messages))
        return super().complete(messages)


def last_text(messages: Sequence[BaseMessage]) -> str:
    return message_text(messages[-1])
