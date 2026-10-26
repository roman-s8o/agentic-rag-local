# ADR 0002: OpenAI-compatible provider abstraction

## Status

Accepted. Amended 2026-10-26: the default backend changed from Ollama to vLLM.

## Context

Production inference runs on vLLM on NVIDIA GPUs. The development machine is Apple Silicon
without an NVIDIA GPU. vLLM's Apple Silicon plugin (vllm-metal) focused on text-only models
as of October 2026, and this project needs a vision model, so the Mac uses Ollama or
mlx-vlm. vLLM, Ollama and mlx-vlm all expose `/v1/chat/completions`.

## Decision

All generation, structured output and vision calls go through `LLMProvider`, implemented
by `OpenAICompatibleProvider` (`langchain-openai.ChatOpenAI`). The backend is selected by
`llm.backend` (`vllm` | `ollama` | `mlx` | `openai`). Structured output uses
`json_schema` when the server supports it and falls back to a JSON-in-prompt retry loop
(MLX, older Ollama).

The default backend is `vllm` with `Qwen/Qwen3-VL-8B-Instruct`, because that is how the
system runs on a GPU server. Ollama and MLX are the Mac fallbacks.

vLLM is a `gpu` compose profile with a pinned image tag (`VLLM_IMAGE_TAG`), not a code path.

## Consequences

- Application code, tests and the agent graph never import Ollama, MLX or vLLM SDKs.
- Switching hardware is an env var (`RAG_LLM__BACKEND`, `RAG_LLM__BASE_URL`, `RAG_LLM__MODEL`).
- The default quickstart needs an NVIDIA GPU; on a Mac it takes three env vars.
- Features that are not in the OpenAI chat schema (native tool calling quirks, custom
  sampling) are out of scope. That is a deliberate trade for portability.
