import math
from pathlib import Path
from typing import Any

from agentic_rag.evaluation import (
    EvalRecord,
    GoldenSample,
    check_thresholds,
    load_dataset,
    render_markdown,
    run_evaluation,
    source_recall,
    summarize,
    write_report,
)
from agentic_rag.schema import Chunk, ScoredChunk
from agentic_rag.services import AgentResult


def _record(**scores: float) -> EvalRecord:
    return EvalRecord(
        question="q",
        reference="r",
        response="a",
        contexts=["c"],
        image_contexts=[],
        retrieved_sources=["a.md"],
        reference_sources=["a.md", "b.md"],
        grounded=True,
        trace_id=None,
        scores=dict(scores),
    )


def test_summarize_ignores_nan() -> None:
    summary = summarize([_record(faithfulness=1.0), _record(faithfulness=math.nan)])
    assert summary == {"faithfulness": 1.0}


def test_threshold_gate() -> None:
    failures = check_thresholds(
        {"faithfulness": 0.9, "context_recall": 0.5},
        {"faithfulness": 0.8, "context_recall": 0.7, "answer_relevancy": 0.7},
    )
    assert set(failures) == {"context_recall", "answer_relevancy"}
    assert failures["context_recall"] == (0.5, 0.7)


def test_source_recall() -> None:
    assert source_recall(_record()) == 0.5


def test_project_dataset_is_valid() -> None:
    samples = load_dataset(Path("eval/datasets/golden.jsonl"))
    assert len(samples) >= 10
    corpus = {p.name for p in Path("eval/corpus").iterdir()}
    assert all(set(s.reference_sources) <= corpus for s in samples)


class FakeServices:
    settings: Any = None

    def __init__(self) -> None:
        from agentic_rag.config import ObservabilitySettings, Settings

        self.settings = Settings(observability=ObservabilitySettings(langfuse_enabled=False))

    def ask(self, question: str, **_: Any) -> AgentResult:
        doc = ScoredChunk(chunk=Chunk(id="1", text="98.2 %", source="a.md"), score=1.0)
        return AgentResult(
            answer="98.2 % [1]",
            citations=[],
            grounded=True,
            route="retrieve",
            queries=[question],
            thread_id="t",
            state={"documents": [doc]},
        )


def test_run_evaluation_and_report(tmp_path: Path) -> None:
    samples = [GoldenSample(question="q", reference="r", reference_sources=["a.md"])]
    report = run_evaluation(
        FakeServices(),  # type: ignore[arg-type]
        samples,
        scorer=lambda r: {"faithfulness": 1.0 if r.contexts else 0.0},
        thresholds={"faithfulness": 0.8, "source_recall": 1.0},
    )
    assert report.passed
    assert report.summary == {"faithfulness": 1.0, "source_recall": 1.0}
    md = write_report(report, {"faithfulness": 0.8}, tmp_path)
    assert "| faithfulness | 1.000 | 0.80 | PASS |" in md.read_text()
    assert (tmp_path / "latest.json").exists()
    assert "source_recall" in render_markdown(report, {})
