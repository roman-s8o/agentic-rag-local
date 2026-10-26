# Changelog

## Unreleased

- Default LLM backend is vLLM with `Qwen/Qwen3-VL-8B-Instruct` (was Ollama); Ollama and MLX remain for Macs ([ADR 0002](docs/adr/0002-provider-abstraction.md))
- vLLM image pinned to `v0.31.0` (override with `VLLM_IMAGE_TAG`); the `api` container points at the `vllm` service by default
- README: evaluation status and CI scope described as they are

## 0.1.0 - 2026-10-25

Initial public release.

- Hugging Face downloader (`rag models pull` / `pull-all`) with license/gating checks and optional MLX conversion
- OpenAI-compatible LLM provider (Ollama / MLX / vLLM / OpenAI) with structured-output fallback
- Docling ingestion for text and images; BGE-M3 dense + sparse embeddings
- Qdrant hybrid search (RRF) and ColQwen late-interaction image vectors
- Cross-encoder reranker (or HTTP / no-op)
- LangGraph agent: route, retrieve, rerank, grade, rewrite, generate, groundedness check
- Langfuse tracing and RAGAS evaluation with CI threshold gate
- FastAPI (`/query`, `/query/stream`, `/ingest`) and Gradio UI
