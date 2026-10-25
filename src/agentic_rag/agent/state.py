from __future__ import annotations

from typing import Annotated, Any, Literal, Required, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from agentic_rag.schema import Citation, ScoredChunk

Route = Literal["direct", "retrieve", "decompose"]


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    question: Required[str]
    query_image: str | None
    sources: list[str] | None
    route: Route
    standalone_question: str
    queries: list[str]
    candidates: list[ScoredChunk]
    documents: list[ScoredChunk]
    retrieval_rounds: int
    generation_attempts: int
    missing_info: str | None
    answer: str
    citations: list[Citation]
    grounded: bool | None
    feedback: str | None


def standalone(state: AgentState) -> str:
    return state.get("standalone_question") or state["question"]


def initial_state(
    question: str, query_image: str | None = None, sources: list[str] | None = None
) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "query_image": query_image,
        "sources": sources,
        "queries": [],
        "candidates": [],
        "documents": [],
        "retrieval_rounds": 0,
        "generation_attempts": 0,
        "missing_info": None,
        "answer": "",
        "citations": [],
        "grounded": None,
        "feedback": None,
    }


class QueryAnalysis(BaseModel):
    route: Route = Field(
        description="direct: greetings or questions about the conversation itself; "
        "retrieve: a single factual question; decompose: a multi-part or comparative question."
    )
    standalone_question: str = Field(
        description="The question rewritten to be self-contained using the chat history."
    )
    sub_queries: list[str] = Field(
        default_factory=list, description="Search queries for each part when route=decompose."
    )


class RelevanceGrades(BaseModel):
    relevant_ids: list[int] = Field(
        default_factory=list, description="Numbers of the passages that help answer the question."
    )
    sufficient: bool = Field(
        description="True if the relevant passages together fully answer the question."
    )
    missing_info: str = Field(default="", description="What information is still missing.")


class RewrittenQueries(BaseModel):
    queries: list[str] = Field(description="1-3 new search queries targeting the missing info.")


class GroundednessVerdict(BaseModel):
    grounded: bool = Field(description="True if every claim in the answer is supported.")
    unsupported_claims: list[str] = Field(default_factory=list)
