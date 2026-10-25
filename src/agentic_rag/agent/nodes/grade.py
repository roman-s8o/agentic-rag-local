from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.nodes.context import format_passages, merge_documents
from agentic_rag.agent.prompts import GRADE_SYSTEM
from agentic_rag.agent.state import AgentState, RelevanceGrades, standalone


class GradeDocuments:
    """CRAG-style relevance filter that also judges whether the evidence is sufficient."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        candidates = state.get("candidates", [])
        if not candidates:
            return {"missing_info": "No passages were retrieved."}
        passages = format_passages(candidates, self.deps.settings.context_chars_per_doc)
        grades = self.deps.llm.structured(
            [
                SystemMessage(content=GRADE_SYSTEM),
                HumanMessage(content=f"Question: {standalone(state)}\n\nPassages:\n{passages}"),
            ],
            RelevanceGrades,
        )
        relevant = [candidates[i - 1] for i in grades.relevant_ids if 1 <= i <= len(candidates)]
        documents = merge_documents(state.get("documents", []), relevant)
        missing = None if grades.sufficient and documents else (grades.missing_info or "unknown")
        return {"documents": documents, "missing_info": missing}


def route_after_grade(state: AgentState, max_rounds: int) -> Literal["generate", "rewrite_query"]:
    if state.get("missing_info") and state.get("retrieval_rounds", 0) < max_rounds:
        return "rewrite_query"
    return "generate"
