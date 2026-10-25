from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from langchain_core.messages import AnyMessage, BaseMessage

from agentic_rag.schema import Citation, ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+)\]")


def format_passages(docs: Sequence[ScoredChunk], max_chars: int) -> str:
    return "\n\n".join(
        f"[{i}] ({d.chunk.label()})\n{d.chunk.text[:max_chars]}" for i, d in enumerate(docs, 1)
    )


def context_images(docs: Sequence[ScoredChunk], limit: int) -> list[str]:
    paths = [
        d.chunk.image_path
        for d in docs
        if d.chunk.modality == "image" and d.chunk.image_path and Path(d.chunk.image_path).exists()
    ]
    return paths[:limit]


def history(messages: Sequence[AnyMessage], limit: int = 6) -> list[BaseMessage]:
    return list(messages[:-1][-limit:])


def extract_citations(answer: str, docs: Sequence[ScoredChunk]) -> list[Citation]:
    seen: dict[int, Citation] = {}
    for match in _CITATION_RE.finditer(answer):
        idx = int(match.group(1))
        if idx in seen or not 1 <= idx <= len(docs):
            continue
        chunk = docs[idx - 1].chunk
        seen[idx] = Citation(
            index=idx,
            source=chunk.source,
            page=chunk.page,
            modality=chunk.modality,
            image_path=chunk.image_path,
            snippet=chunk.text[:240],
        )
    return list(seen.values())


def merge_documents(
    existing: Sequence[ScoredChunk], new: Sequence[ScoredChunk]
) -> list[ScoredChunk]:
    merged = {d.chunk.id: d for d in existing}
    for d in new:
        merged.setdefault(d.chunk.id, d)
    return sorted(
        merged.values(),
        key=lambda d: d.rerank_score if d.rerank_score is not None else d.score,
        reverse=True,
    )
