# ADR 0003: Hybrid search with BGE-M3 and RRF

## Status

Accepted.

## Context

Lexical and semantic retrieval fail on different queries. A learned sparse model
(BGE-M3) outperforms BM25 on out-of-domain text and shares a forward pass with the dense
encoder. Image pages need a third signal.

## Decision

- Default text encoder: BGE-M3 (`text_backend: bgem3`) producing dense + sparse together.
- Fallback: any sentence-transformers model + FastEmbed BM25 (`text_backend: sentence_transformers`).
- Qdrant Query API prefetches each named vector, then fuses with Reciprocal Rank Fusion.
- Multiple rewritten / decomposed queries are fused again client-side with the same RRF.
- A cross-encoder reranker then keeps `reranker.top_n` passages.

## Consequences

- One embedding model download covers both channels for text.
- BM25 is still available when BGE-M3 is too heavy (CI, tiny machines).
- Image retrieval is optional (`embedding.image_enabled`); when off, the collection is
  created without the `image` vector so older indexes stay valid.
