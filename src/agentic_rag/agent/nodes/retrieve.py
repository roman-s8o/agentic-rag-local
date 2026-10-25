from __future__ import annotations

from typing import Any

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.state import AgentState, standalone


class Retrieve:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidates = self.deps.retriever.retrieve_many(
            state.get("queries", []), state.get("sources")
        )
        return {
            "candidates": candidates,
            "retrieval_rounds": state.get("retrieval_rounds", 0) + 1,
        }


class Rerank:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        reranked = self.deps.reranker.rerank(
            standalone(state), state.get("candidates", []), self.deps.rerank_top_n
        )
        return {"candidates": reranked}
