from pathlib import Path

from agentic_rag.ingestion.pipeline import IngestionPipeline
from agentic_rag.retrieval import HybridRetriever, NoopReranker, reciprocal_rank_fusion
from agentic_rag.schema import Chunk, ScoredChunk
from tests.fakes import KeywordReranker


def _sc(cid: str, score: float = 0.0) -> ScoredChunk:
    return ScoredChunk(chunk=Chunk(id=cid, text=cid, source="s"), score=score)


def test_rrf_rewards_consensus() -> None:
    fused = reciprocal_rank_fusion([[_sc("a"), _sc("b"), _sc("c")], [_sc("b"), _sc("c")]], k=60)
    assert [d.chunk.id for d in fused] == ["b", "c", "a"]
    assert fused[0].score == 1 / 61 + 1 / 62


def test_rrf_limit() -> None:
    assert len(reciprocal_rank_fusion([[_sc("a"), _sc("b")]], limit=1)) == 1


def test_hybrid_search_finds_relevant_chunk(
    pipeline: IngestionPipeline, retriever: HybridRetriever, corpus: Path
) -> None:
    pipeline.ingest(corpus)
    results = retriever.retrieve("What is the peak efficiency of the Helios inverter?")
    assert results
    assert results[0].chunk.source == "inverter.txt"


def test_source_filter(
    pipeline: IngestionPipeline, retriever: HybridRetriever, corpus: Path
) -> None:
    pipeline.ingest(corpus)
    results = retriever.retrieve("hotel night cap", sources=["policy.txt"])
    assert results
    assert {r.chunk.source for r in results} == {"policy.txt"}


def test_retrieve_many_dedupes(
    pipeline: IngestionPipeline, retriever: HybridRetriever, corpus: Path
) -> None:
    pipeline.ingest(corpus)
    results = retriever.retrieve_many(["inverter efficiency", "hotel cap per night"])
    ids = [r.chunk.id for r in results]
    assert len(ids) == len(set(ids))
    assert {"inverter.txt", "policy.txt"} <= {r.chunk.source for r in results}


def test_rerankers() -> None:
    docs = [_sc("hotel policy"), _sc("inverter efficiency")]
    top = KeywordReranker().rerank("inverter", docs, top_n=1)
    assert top[0].chunk.id == "inverter efficiency"
    assert top[0].rerank_score == 1.0
    assert NoopReranker().rerank("q", docs, top_n=1) == docs[:1]
