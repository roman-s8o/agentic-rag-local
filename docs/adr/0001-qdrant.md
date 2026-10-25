# ADR 0001: Qdrant as the vector store

## Status

Accepted.

## Context

The retrieval layer must do hybrid search (dense + sparse) and store ColQwen-style
multivectors for images. Candidates were Qdrant, Postgres + pgvector + ParadeDB, and
Weaviate.

## Decision

Use Qdrant.

## Consequences

- Named vectors (`dense`, `sparse`, `image`) live in one collection; RRF is a first-class
  Query API fusion, not a client-side merge.
- Multivector + MaxSim is native, so ColQwen does not need a second store.
- Local mode (`:memory:` / on-disk) is enough for unit tests; the same client talks to the
  Docker service in integration tests and production.
- Operational cost is one extra container. Postgres-only shops lose a single-process
  deployment, which we judged acceptable for a local-first stack.
