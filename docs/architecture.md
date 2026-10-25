# Architecture

The system is a local multimodal agentic RAG stack. Every inference call goes through an
OpenAI-compatible HTTP API, so the same application code runs on Ollama or MLX (this Mac)
and on vLLM (NVIDIA GPU).

```
PDF / DOCX / images
        |
     Docling + HybridChunker
        |
   BGE-M3 (dense + sparse)     ColQwen (image multivectors)
        |                               |
        +------------ Qdrant -----------+
                        |
        query --> LangGraph agent
                        |
          analyze --> retrieve -- rerank -- grade
                         ^                    |
                         +---- rewrite -------+
                                              v
                                    generate --> groundedness
                                              |
                                         citations
```

## Layers

- **Ingestion.** [Docling](https://github.com/DS4SD/docling) parses PDFs and office files,
  extracts pictures and uses HybridChunker (token-bounded, heading-aware). Images below
  `min_image_px` are dropped. Remaining images are captioned by the VLM and stored under
  `data/images/`.
- **Embeddings.** BGE-M3 produces a 1024-d dense vector and a learned sparse lexical vector
  in one pass. ColQwen2.5 produces late-interaction multivectors for page/image retrieval.
- **Store.** One Qdrant collection with named vectors `dense`, `sparse` and `image`.
  Hybrid search is Qdrant's Query API with Reciprocal Rank Fusion (RRF).
- **Reranker.** Cross-encoder (`BAAI/bge-reranker-v2-m3`) by default. An HTTP backend
  talks to any Cohere/Jina-compatible `/rerank` endpoint (vLLM, Infinity, TEI).
- **Agent.** LangGraph graph: analyze, retrieve, rerank, grade, rewrite, generate, check.
  SQLite checkpointer keeps multi-turn state. See [ADR 0004](adr/0004-agent-loop.md).
- **Observability.** Langfuse CallbackHandler on every graph run. Scores from RAGAS are
  attached to the same traces.
- **Evaluation.** `eval/datasets/golden.jsonl` is the golden set. `rag eval` runs the
  agent, scores with RAGAS (faithfulness, answer relevancy, context precision/recall,
  multimodal faithfulness) and fails if any configured threshold is missed.

## Default models (8-16 GB friendly)

| Role | Model | Notes |
|---|---|---|
| Generator | `qwen3-vl:8b` (Ollama) / `Qwen/Qwen3-VL-8B-Instruct` (vLLM) | VLM |
| Embeddings | `BAAI/bge-m3` | dense + sparse |
| Reranker | `BAAI/bge-reranker-v2-m3` | cross-encoder |
| Images | `vidore/colqwen2.5-v0.2` | late interaction |

Swap any of them in `configs/models.yaml` and `configs/settings.yaml`.
