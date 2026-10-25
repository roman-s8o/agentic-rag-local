from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, SystemMessage

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.nodes.context import (
    context_images,
    extract_citations,
    format_passages,
    history,
)
from agentic_rag.agent.prompts import DIRECT_SYSTEM, GENERATE_SYSTEM, NO_CONTEXT_ANSWER
from agentic_rag.agent.state import AgentState, standalone
from agentic_rag.providers import user_message


class Generate:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        attempts = state.get("generation_attempts", 0) + 1
        docs = state.get("documents", [])
        if not docs:
            return {
                "answer": NO_CONTEXT_ANSWER,
                "citations": [],
                "grounded": True,
                "generation_attempts": attempts,
            }
        cfg = self.deps.settings
        images: list[str] = context_images(docs, cfg.max_context_images)
        if query_image := state.get("query_image"):
            images.append(query_image)
        prompt = (
            f"Context:\n{format_passages(docs, cfg.context_chars_per_doc)}\n\n"
            f"Question: {standalone(state)}"
        )
        if feedback := state.get("feedback"):
            prompt += f"\n\nYour previous answer had unsupported claims: {feedback}. Fix them."
        answer = self.deps.llm.complete(
            [
                SystemMessage(content=GENERATE_SYSTEM),
                *history(state.get("messages", [])),
                user_message(prompt, list(images) or None),
            ]
        )
        return {
            "answer": answer,
            "citations": extract_citations(answer, docs),
            "generation_attempts": attempts,
        }


class DirectAnswer:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        answer = self.deps.llm.complete(
            [
                SystemMessage(content=DIRECT_SYSTEM),
                *history(state.get("messages", [])),
                user_message(state["question"]),
            ]
        )
        return {"answer": answer, "citations": [], "grounded": True}


def finalize(state: AgentState) -> dict[str, Any]:
    return {"messages": [AIMessage(content=state.get("answer", ""))]}
