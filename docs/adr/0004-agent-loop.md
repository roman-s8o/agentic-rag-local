# ADR 0004: Corrective + Self-RAG agent loop

## Status

Accepted.

## Context

Naive retrieve-then-generate hallucinates when the top-k is wrong and cannot recover.
Full ReAct tool-calling agents are harder to evaluate and to bound.

## Decision

A typed LangGraph state machine with hard iteration caps:

1. **Analyze** — route to `direct`, `retrieve` or `decompose`; rewrite a standalone question.
2. **Retrieve + rerank** — hybrid search, then cross-encoder.
3. **Grade** (CRAG) — keep relevant passages; if insufficient and rounds remain, rewrite.
4. **Generate** — VLM answer with numbered citations; attach retrieved images.
5. **Check** (Self-RAG) — groundedness verdict; one regeneration if unsupported claims remain.

`max_retrieval_rounds` and `max_generation_attempts` (default 2) make the graph finite.
SQLite checkpointer persists `messages` across turns.

## Consequences

- Every decision is a Pydantic structured call, so the loop is testable with a scripted LLM
  (see `tests/test_agent.py`).
- Cost is several extra LLM calls per question. That is acceptable for a local 8B VLM and
  is the point of the project: show the loops, not hide them.
- The graph does not browse the web and does not call arbitrary tools.
