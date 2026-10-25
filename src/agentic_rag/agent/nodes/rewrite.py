from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.prompts import REWRITE_SYSTEM
from agentic_rag.agent.state import AgentState, RewrittenQueries, standalone


class RewriteQuery:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        previous = state.get("queries", [])
        result = self.deps.llm.structured(
            [
                SystemMessage(content=REWRITE_SYSTEM),
                HumanMessage(
                    content=(
                        f"Question: {standalone(state)}\n"
                        f"Previous queries: {previous}\n"
                        f"Missing information: {state.get('missing_info') or 'unknown'}"
                    )
                ),
            ],
            RewrittenQueries,
        )
        seen = {q.lower() for q in previous}
        queries = [q for q in result.queries if q.strip() and q.lower() not in seen]
        return {"queries": queries[:3] or [standalone(state)]}
