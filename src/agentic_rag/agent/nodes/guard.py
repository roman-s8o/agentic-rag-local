from __future__ import annotations

from typing import Any, Literal

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.prompts import BLOCKED_ANSWER
from agentic_rag.agent.state import AgentState


class ScreenInput:
    """Optional: stop a prompt injection in the user's message before any model sees it."""

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        if self.deps.guard is None:
            return {}
        if self.deps.guard.check_user_message(state["question"]).ok:
            return {}
        return {"blocked": True, "answer": BLOCKED_ANSWER, "citations": [], "grounded": None}


class ScreenDocuments:
    """Optional: drop retrieved chunks that carry instructions for the model (indirect injection).

    Only the number of dropped chunks is kept in the state; chunk text is never logged here.
    """

    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps

    def __call__(self, state: AgentState) -> dict[str, Any]:
        guard = self.deps.guard
        if guard is None:
            return {}
        candidates = state.get("candidates", [])
        kept = [c for c in candidates if guard.check_document(c.chunk.text).ok]
        return {
            "candidates": kept,
            "guard_dropped": state.get("guard_dropped", 0) + len(candidates) - len(kept),
        }


def route_after_screen(state: AgentState) -> Literal["analyze", "finalize"]:
    return "finalize" if state.get("blocked") else "analyze"
