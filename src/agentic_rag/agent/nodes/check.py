from __future__ import annotations

from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.nodes.context import context_images, format_passages
from agentic_rag.agent.prompts import CHECK_SYSTEM
from agentic_rag.agent.state import AgentState, GroundednessVerdict
from agentic_rag.providers import user_message


class CheckGroundedness:
    """Self-RAG style hallucination check of the generated answer against its context."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        docs = state.get("documents", [])
        if not docs:
            return {"grounded": True, "feedback": None}
        cfg = self.deps.settings
        prompt = (
            f"Context:\n{format_passages(docs, cfg.context_chars_per_doc)}\n\n"
            f"Answer to verify:\n{state.get('answer', '')}"
        )
        images = context_images(docs, cfg.max_context_images)
        verdict = self.deps.llm.structured(
            [
                SystemMessage(content=CHECK_SYSTEM),
                user_message(prompt, list(images)) if images else HumanMessage(content=prompt),
            ],
            GroundednessVerdict,
        )
        feedback = "; ".join(verdict.unsupported_claims) or None
        return {"grounded": verdict.grounded, "feedback": None if verdict.grounded else feedback}


def route_after_check(state: AgentState, max_attempts: int) -> Literal["generate", "finalize"]:
    if state.get("grounded") is False and state.get("generation_attempts", 0) < max_attempts:
        return "generate"
    return "finalize"
