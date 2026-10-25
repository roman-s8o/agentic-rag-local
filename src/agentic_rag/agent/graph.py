from __future__ import annotations

import sqlite3
from functools import partial
from pathlib import Path
from typing import Literal

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.agent.deps import AgentDeps
from agentic_rag.agent.nodes import (
    AnalyzeQuery,
    CheckGroundedness,
    DirectAnswer,
    Generate,
    GradeDocuments,
    Rerank,
    Retrieve,
    RewriteQuery,
    finalize,
    route_after_check,
    route_after_grade,
)
from agentic_rag.agent.state import AgentState


def route_after_analyze(state: AgentState) -> Literal["direct_answer", "retrieve"]:
    return "direct_answer" if state.get("route") == "direct" else "retrieve"


def build_graph(
    deps: AgentDeps, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    cfg = deps.settings
    graph = StateGraph(AgentState)
    graph.add_node("analyze", AnalyzeQuery(deps))
    graph.add_node("direct_answer", DirectAnswer(deps))
    graph.add_node("retrieve", Retrieve(deps))
    graph.add_node("rerank", Rerank(deps))
    graph.add_node("grade", GradeDocuments(deps))
    graph.add_node("rewrite_query", RewriteQuery(deps))
    graph.add_node("generate", Generate(deps))
    graph.add_node("check", CheckGroundedness(deps))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "analyze")
    graph.add_conditional_edges("analyze", route_after_analyze, ["direct_answer", "retrieve"])
    graph.add_edge("retrieve", "rerank")
    graph.add_edge("rerank", "grade")
    graph.add_conditional_edges(
        "grade",
        partial(route_after_grade, max_rounds=cfg.max_retrieval_rounds),
        ["generate", "rewrite_query"],
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate", "check")
    graph.add_conditional_edges(
        "check",
        partial(route_after_check, max_attempts=cfg.max_generation_attempts),
        ["generate", "finalize"],
    )
    graph.add_edge("direct_answer", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer, name="agentic-rag")


def sqlite_checkpointer(path: Path) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)
