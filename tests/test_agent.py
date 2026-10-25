from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel

from agentic_rag.agent import AgentDeps, build_graph, initial_state, sqlite_checkpointer
from agentic_rag.agent.prompts import NO_CONTEXT_ANSWER
from agentic_rag.agent.state import (
    GroundednessVerdict,
    QueryAnalysis,
    RelevanceGrades,
    RewrittenQueries,
)
from agentic_rag.config import AgentSettings
from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever
from tests.fakes import KeywordReranker, ScriptedLLM


def seq(*items: BaseModel) -> Any:
    it: Iterator[BaseModel] = iter(items)
    last: list[BaseModel] = [items[0]]

    def handler(_: Any) -> BaseModel:
        last[0] = next(it, last[0])
        return last[0]

    return handler


RETRIEVE = QueryAnalysis(
    route="retrieve", standalone_question="What is the peak efficiency of the Helios H5 inverter?"
)
GOOD = RelevanceGrades(relevant_ids=[1], sufficient=True)
GROUNDED = GroundednessVerdict(grounded=True)


@pytest.fixture
def indexed(pipeline: IngestionPipeline, corpus: Path) -> None:
    pipeline.ingest(corpus)


def make_graph(
    llm: ScriptedLLM, retriever: HybridRetriever, checkpointer: Any = None, **settings: Any
) -> Any:
    deps = AgentDeps(
        llm=llm,
        retriever=retriever,
        reranker=KeywordReranker(),
        settings=AgentSettings(**settings),
        rerank_top_n=3,
    )
    return build_graph(deps, checkpointer)


@pytest.mark.usefixtures("indexed")
def test_happy_path_with_citations(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(GOOD),
            "GroundednessVerdict": seq(GROUNDED),
        },
        responses=["Peak efficiency is 98.2 percent [1]."],
    )
    out = make_graph(llm, retriever).invoke(initial_state("helios efficiency?"))
    assert out["route"] == "retrieve"
    assert out["retrieval_rounds"] == 1
    assert out["grounded"] is True
    assert out["answer"].startswith("Peak efficiency")
    assert [c.source for c in out["citations"]] == ["inverter.txt"]
    assert out["messages"][-1].content == out["answer"]


@pytest.mark.usefixtures("indexed")
def test_rewrite_loop_when_evidence_insufficient(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(
                RelevanceGrades(relevant_ids=[], sufficient=False, missing_info="efficiency"),
                GOOD,
            ),
            "RewrittenQueries": seq(RewrittenQueries(queries=["Helios H5 peak efficiency"])),
            "GroundednessVerdict": seq(GROUNDED),
        },
        responses=["98.2 percent [1]."],
    )
    out = make_graph(llm, retriever, max_retrieval_rounds=3).invoke(initial_state("q"))
    assert out["retrieval_rounds"] == 2
    assert out["queries"] == ["Helios H5 peak efficiency"]
    assert [name for name, _ in llm.calls].count("RewrittenQueries") == 1


@pytest.mark.usefixtures("indexed")
def test_no_relevant_documents_falls_back(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(RelevanceGrades(relevant_ids=[], sufficient=False)),
            "RewrittenQueries": seq(RewrittenQueries(queries=["other"])),
        }
    )
    out = make_graph(llm, retriever, max_retrieval_rounds=2).invoke(initial_state("q"))
    assert out["retrieval_rounds"] == 2
    assert out["answer"] == NO_CONTEXT_ANSWER
    assert out["citations"] == []


@pytest.mark.usefixtures("indexed")
def test_regenerates_when_not_grounded(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(GOOD),
            "GroundednessVerdict": seq(
                GroundednessVerdict(grounded=False, unsupported_claims=["99 percent"]),
                GROUNDED,
            ),
        },
        responses=["It is 99 percent [1].", "It is 98.2 percent [1]."],
    )
    out = make_graph(llm, retriever).invoke(initial_state("q"))
    assert out["generation_attempts"] == 2
    assert out["grounded"] is True
    assert out["answer"] == "It is 98.2 percent [1]."


@pytest.mark.usefixtures("indexed")
def test_generation_attempts_are_capped(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(GOOD),
            "GroundednessVerdict": seq(GroundednessVerdict(grounded=False)),
        },
        responses=["a [1]", "b [1]", "c [1]"],
    )
    out = make_graph(llm, retriever, max_generation_attempts=2).invoke(initial_state("q"))
    assert out["generation_attempts"] == 2
    assert out["grounded"] is False


def test_direct_route_skips_retrieval(retriever: HybridRetriever) -> None:
    llm = ScriptedLLM(
        {"QueryAnalysis": seq(QueryAnalysis(route="direct", standalone_question="hi"))},
        responses=["Hello!"],
    )
    out = make_graph(llm, retriever).invoke(initial_state("hi"))
    assert out["answer"] == "Hello!"
    assert out["retrieval_rounds"] == 0


@pytest.mark.usefixtures("indexed")
def test_decompose_uses_sub_queries(retriever: HybridRetriever) -> None:
    analysis = QueryAnalysis(
        route="decompose",
        standalone_question="Compare inverter voltage and hotel cap",
        sub_queries=["inverter max input voltage", "hotel cap per night"],
    )
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(analysis),
            "RelevanceGrades": seq(RelevanceGrades(relevant_ids=[1, 2], sufficient=True)),
            "GroundednessVerdict": seq(GROUNDED),
        },
        responses=["600 V [1], 180 EUR [2]."],
    )
    out = make_graph(llm, retriever).invoke(initial_state("compare"))
    assert out["queries"][:2] == analysis.sub_queries
    assert {c.source for c in out["citations"]} == {"inverter.txt", "policy.txt"}


@pytest.mark.usefixtures("indexed")
def test_checkpointer_keeps_conversation(retriever: HybridRetriever, tmp_path: Path) -> None:
    llm = ScriptedLLM(
        {
            "QueryAnalysis": seq(RETRIEVE),
            "RelevanceGrades": seq(GOOD),
            "GroundednessVerdict": seq(GROUNDED),
        },
        responses=["98.2 percent [1]."],
    )
    graph = make_graph(llm, retriever, sqlite_checkpointer(tmp_path / "cp.sqlite"))
    config = {"configurable": {"thread_id": "t1"}}
    graph.invoke(initial_state("first"), config)
    out = graph.invoke(initial_state("second"), config)
    assert [m.content for m in out["messages"]][::2] == ["first", "second"]
    assert out["retrieval_rounds"] == 1
