from __future__ import annotations

from typing import Any

from langchain_core.messages import SystemMessage

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.nodes.context import history
from agentic_rag.agent.prompts import ANALYZE_SYSTEM
from agentic_rag.agent.state import AgentState, QueryAnalysis
from agentic_rag.providers import user_message


class AnalyzeQuery:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        max_sub = self.deps.settings.max_sub_queries
        question = state["question"]
        image = state.get("query_image")
        analysis = self.deps.llm.structured(
            [
                SystemMessage(content=ANALYZE_SYSTEM.format(max_sub=max_sub)),
                *history(state.get("messages", [])),
                user_message(question, [image] if image else None),
            ],
            QueryAnalysis,
        )
        standalone = analysis.standalone_question.strip() or question
        queries = [standalone]
        if analysis.route == "decompose":
            queries = [*analysis.sub_queries[:max_sub], standalone]
        return {"route": analysis.route, "standalone_question": standalone, "queries": queries}
