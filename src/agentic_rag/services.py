from __future__ import annotations

import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agentic_rag import observability
from agentic_rag.agent import AgentDeps, build_graph, initial_state, sqlite_checkpointer
from agentic_rag.config import Settings
from agentic_rag.embeddings import (
    ImageEmbedder,
    TextEncoder,
    build_image_embedder,
    build_text_encoder,
)
from agentic_rag.ingestion import DoclingParser, ImageCaptioner, IngestionPipeline
from agentic_rag.providers import LLMProvider, build_llm
from agentic_rag.retrieval import HybridRetriever, QdrantStore, Reranker, build_reranker
from agentic_rag.schema import Citation


@dataclass
class AgentResult:
    answer: str
    citations: list[Citation]
    grounded: bool | None
    route: str | None
    queries: list[str]
    thread_id: str
    trace_id: str | None = None
    state: dict[str, Any] = field(default_factory=dict, repr=False)


class Services:
    """Composition root: builds every component lazily from settings."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @cached_property
    def llm(self) -> LLMProvider:
        return build_llm(self.settings.llm)

    @cached_property
    def store(self) -> QdrantStore:
        return QdrantStore.from_settings(self.settings.qdrant)

    @cached_property
    def text_encoder(self) -> TextEncoder:
        return build_text_encoder(self.settings.embedding)

    @cached_property
    def image_embedder(self) -> ImageEmbedder | None:
        return build_image_embedder(self.settings.embedding)

    @cached_property
    def reranker(self) -> Reranker:
        return build_reranker(self.settings.reranker, self.settings.embedding.device)

    @cached_property
    def retriever(self) -> HybridRetriever:
        return HybridRetriever(
            self.store, self.text_encoder, self.settings.retrieval, self.image_embedder
        )

    @cached_property
    def pipeline(self) -> IngestionPipeline:
        s = self.settings
        return IngestionPipeline(
            store=self.store,
            text_encoder=self.text_encoder,
            settings=s.ingestion,
            parser=DoclingParser(s.embedding.dense_model, s.ingestion.max_chunk_tokens),
            image_embedder=self.image_embedder,
            captioner=ImageCaptioner(self.llm) if s.ingestion.caption_images else None,
        )

    @cached_property
    def graph(self) -> CompiledStateGraph:
        deps = AgentDeps(
            llm=self.llm,
            retriever=self.retriever,
            reranker=self.reranker,
            settings=self.settings.agent,
            rerank_top_n=self.settings.reranker.top_n,
        )
        return build_graph(deps, sqlite_checkpointer(self.settings.agent.checkpoint_path))

    def _config(self, thread_id: str | None, tags: tuple[str, ...] = ()) -> tuple[Any, Any, str]:
        thread = thread_id or uuid.uuid4().hex
        handler = observability.new_handler(self.settings.observability)
        config = observability.run_config(self.settings.observability, thread, handler, tags=tags)
        return config, handler, thread

    def ask(
        self,
        question: str,
        image: str | None = None,
        thread_id: str | None = None,
        sources: list[str] | None = None,
        tags: tuple[str, ...] = (),
    ) -> AgentResult:
        config, handler, thread = self._config(thread_id, tags)
        state = self.graph.invoke(initial_state(question, image, sources), config)
        observability.flush(self.settings.observability)
        return AgentResult(
            answer=state.get("answer", ""),
            citations=state.get("citations", []),
            grounded=state.get("grounded"),
            route=state.get("route"),
            queries=state.get("queries", []),
            thread_id=thread,
            trace_id=observability.trace_id(handler),
            state=state,
        )

    def stream(
        self,
        question: str,
        image: str | None = None,
        thread_id: str | None = None,
        sources: list[str] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Yields {"type": "node"|"token"|"final", ...} events."""
        config, handler, thread = self._config(thread_id)
        final: dict[str, Any] = {}
        events: Iterator[Any] = self.graph.stream(
            initial_state(question, image, sources),
            config,
            stream_mode=["updates", "messages"],
        )
        for mode, payload in events:
            if mode == "messages":
                chunk, meta = payload
                text = getattr(chunk, "content", "")
                if (
                    meta.get("langgraph_node") in ("generate", "direct_answer")
                    and isinstance(text, str)
                    and text
                ):
                    yield {"type": "token", "text": text}
            elif mode == "updates":
                for node, update in payload.items():
                    final.update(update or {})
                    yield {"type": "node", "node": node}
        observability.flush(self.settings.observability)
        yield {
            "type": "final",
            "answer": final.get("answer", ""),
            "citations": [c.model_dump() for c in final.get("citations", [])],
            "grounded": final.get("grounded"),
            "route": final.get("route"),
            "thread_id": thread,
            "trace_id": observability.trace_id(handler),
        }
