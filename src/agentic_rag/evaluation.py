from __future__ import annotations

import json
import logging
import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from agentic_rag import observability
from agentic_rag.config import LLMSettings, Settings

if TYPE_CHECKING:
    from agentic_rag.embeddings import TextEncoder
    from agentic_rag.services import AgentResult, Services

logger = logging.getLogger(__name__)


class GoldenSample(BaseModel):
    question: str
    reference: str
    image: str | None = None
    reference_sources: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


@dataclass
class EvalRecord:
    question: str
    reference: str
    response: str
    contexts: list[str]
    image_contexts: list[str]
    retrieved_sources: list[str]
    reference_sources: list[str]
    grounded: bool | None
    trace_id: str | None
    scores: dict[str, float] = field(default_factory=dict)


@dataclass
class EvalReport:
    records: list[EvalRecord]
    summary: dict[str, float]
    failures: dict[str, tuple[float, float]]
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def passed(self) -> bool:
        return not self.failures


def load_dataset(path: Path) -> list[GoldenSample]:
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    return [GoldenSample.model_validate_json(line) for line in lines]


def record_from_result(sample: GoldenSample, result: AgentResult) -> EvalRecord:
    docs = result.state.get("documents", [])
    return EvalRecord(
        question=sample.question,
        reference=sample.reference,
        response=result.answer,
        contexts=[d.chunk.text for d in docs],
        image_contexts=[d.chunk.image_path for d in docs if d.chunk.image_path],
        retrieved_sources=list(dict.fromkeys(d.chunk.source for d in docs)),
        reference_sources=sample.reference_sources,
        grounded=result.grounded,
        trace_id=result.trace_id,
    )


def source_recall(record: EvalRecord) -> float | None:
    if not record.reference_sources:
        return None
    hits = sum(1 for s in record.reference_sources if s in record.retrieved_sources)
    return hits / len(record.reference_sources)


def summarize(records: Sequence[EvalRecord]) -> dict[str, float]:
    names = sorted({name for r in records for name in r.scores})
    summary: dict[str, float] = {}
    for name in names:
        values = [r.scores[name] for r in records if not math.isnan(r.scores.get(name, math.nan))]
        summary[name] = statistics.fmean(values) if values else math.nan
    return summary


def check_thresholds(
    summary: dict[str, float], thresholds: dict[str, float]
) -> dict[str, tuple[float, float]]:
    failures: dict[str, tuple[float, float]] = {}
    for name, minimum in thresholds.items():
        value = summary.get(name, math.nan)
        if math.isnan(value) or value < minimum:
            failures[name] = (value, minimum)
    return failures


Scorer = Callable[[EvalRecord], dict[str, float]]


class RagasScorer:
    """Scores records with RAGAS collection metrics using an OpenAI-compatible judge."""

    def __init__(self, judge: LLMSettings, encoder: TextEncoder) -> None:
        from openai import OpenAI
        from ragas.llms import llm_factory
        from ragas.metrics.collections import (
            AnswerRelevancy,
            ContextPrecisionWithReference,
            ContextRecall,
            Faithfulness,
            MultiModalFaithfulness,
        )

        client = OpenAI(
            base_url=judge.resolved_base_url,
            api_key=judge.api_key.get_secret_value(),
            timeout=judge.timeout_s,
        )
        llm: Any = llm_factory(judge.model, client=client)
        embeddings = _ragas_embeddings(encoder)
        self.faithfulness = Faithfulness(llm=llm)
        self.answer_relevancy = AnswerRelevancy(llm=llm, embeddings=embeddings)
        self.context_precision = ContextPrecisionWithReference(llm=llm)
        self.context_recall = ContextRecall(llm=llm)
        self.multimodal_faithfulness = MultiModalFaithfulness(llm=llm)

    def __call__(self, r: EvalRecord) -> dict[str, float]:
        scores: dict[str, float] = {}
        calls: dict[str, Callable[[], Any]] = {
            "faithfulness": lambda: self.faithfulness.score(
                user_input=r.question, response=r.response, retrieved_contexts=r.contexts
            ),
            "answer_relevancy": lambda: self.answer_relevancy.score(
                user_input=r.question, response=r.response
            ),
            "context_precision": lambda: self.context_precision.score(
                user_input=r.question, reference=r.reference, retrieved_contexts=r.contexts
            ),
            "context_recall": lambda: self.context_recall.score(
                user_input=r.question, retrieved_contexts=r.contexts, reference=r.reference
            ),
        }
        if r.image_contexts:
            calls["multimodal_faithfulness"] = lambda: self.multimodal_faithfulness.score(
                response=r.response, retrieved_contexts=[*r.contexts, *r.image_contexts]
            )
        for name, call in calls.items():
            if not r.contexts and name != "answer_relevancy":
                scores[name] = 0.0
                continue
            try:
                scores[name] = float(call().value)
            except Exception as exc:
                logger.warning("Metric %s failed for %r: %s", name, r.question[:60], exc)
                scores[name] = math.nan
        return scores


def _ragas_embeddings(encoder: TextEncoder) -> Any:
    from ragas.embeddings.base import BaseRagasEmbedding

    class EncoderEmbeddings(BaseRagasEmbedding):
        def embed_text(self, text: str, **kwargs: Any) -> list[float]:
            return encoder.encode_query(text).dense

        async def aembed_text(self, text: str, **kwargs: Any) -> list[float]:
            return self.embed_text(text)

    return EncoderEmbeddings()


def run_evaluation(
    services: Services,
    samples: Sequence[GoldenSample],
    scorer: Scorer,
    thresholds: dict[str, float],
) -> EvalReport:
    records: list[EvalRecord] = []
    for i, sample in enumerate(samples, 1):
        logger.info("[%d/%d] %s", i, len(samples), sample.question)
        result = services.ask(sample.question, image=sample.image, tags=("eval",))
        record = record_from_result(sample, result)
        record.scores = scorer(record)
        recall = source_recall(record)
        if recall is not None:
            record.scores["source_recall"] = recall
        for name, value in record.scores.items():
            if not math.isnan(value):
                observability.score(record.trace_id, name, value)
        records.append(record)
    observability.flush(services.settings.observability)
    summary = summarize(records)
    return EvalReport(records, summary, check_thresholds(summary, thresholds))


def render_markdown(report: EvalReport, thresholds: dict[str, float]) -> str:
    lines = [
        f"# RAG evaluation ({report.created_at[:19]}Z, {len(report.records)} samples)",
        "",
        "| Metric | Score | Threshold | Status |",
        "|---|---|---|---|",
    ]
    for name, value in sorted(report.summary.items()):
        minimum = thresholds.get(name)
        status = "-" if minimum is None else ("FAIL" if name in report.failures else "PASS")
        threshold = "-" if minimum is None else f"{minimum:.2f}"
        lines.append(f"| {name} | {value:.3f} | {threshold} | {status} |")
    return "\n".join(lines) + "\n"


def write_report(report: EvalReport, thresholds: dict[str, float], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": report.created_at,
        "passed": report.passed,
        "summary": report.summary,
        "failures": report.failures,
        "records": [asdict(r) for r in report.records],
    }
    (output_dir / "latest.json").write_text(json.dumps(payload, indent=2, default=str))
    md = output_dir / "latest.md"
    md.write_text(render_markdown(report, thresholds))
    return md


def evaluate_from_settings(
    settings: Settings, dataset: Path | None = None, limit: int | None = None
) -> EvalReport:
    from agentic_rag.services import Services

    services = Services(settings)
    samples = load_dataset(dataset or settings.eval.dataset)[:limit]
    scorer = RagasScorer(settings.eval.judge or settings.llm, services.text_encoder)
    return run_evaluation(services, samples, scorer, settings.eval.thresholds)
