from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from agentic_rag.config import AgentSettings
from agentic_rag.providers import LLMProvider
from agentic_rag.retrieval.reranker import Reranker
from agentic_rag.schema import ScoredChunk


class Retriever(Protocol):
    def retrieve_many(
        self, queries: Sequence[str], sources: Sequence[str] | None = None
    ) -> list[ScoredChunk]: ...


class Guard(Protocol):
    """Input/document screening (see the private-rag-guard package: rag_guard.guard.Guard fits)."""

    def check_user_message(self, text: str) -> Any: ...  # result has a boolean .ok
    def check_document(self, text: str) -> Any: ...


@dataclass
class AgentDeps:
    llm: LLMProvider
    retriever: Retriever
    reranker: Reranker
    settings: AgentSettings
    rerank_top_n: int = 6
    guard: Guard | None = None  # None = no screening (default)
