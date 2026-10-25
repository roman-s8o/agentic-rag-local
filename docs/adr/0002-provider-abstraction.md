# ADR 0002: OpenAI-compatible provider abstraction

## Status

Accepted.

## Context

The default development machine is Apple Silicon without an NVIDIA GPU. vLLM is the
production inference engine on CUDA, but it is not usable on this Mac. Ollama and mlx-vlm
both expose `/v1/chat/completions`.

## Decision

All generation, structured output and vision calls go through `LLMProvider`, implemented
by `OpenAICompatibleProvider` (`langchain-openai.ChatOpenAI`). The backend is selected by
`llm.backend` (`ollama` | `mlx` | `vllm` | `openai`). Structured output uses
`json_schema` when the server supports it and falls back to a JSON-in-prompt retry loop
(MLX, older Ollama).

vLLM is a `gpu` compose profile, not a code path.

## Consequences

- Application code, tests and the agent graph never import Ollama, MLX or vLLM SDKs.
- Switching hardware is an env var (`RAG_LLM__BACKEND`, `RAG_LLM__BASE_URL`, `RAG_LLM__MODEL`).
- Features that are not in the OpenAI chat schema (native tool calling quirks, custom
  sampling) are out of scope. That is a deliberate trade for portability.
