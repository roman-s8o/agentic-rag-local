.DEFAULT_GOAL := help
SHELL := /bin/bash

OLLAMA_MODEL ?= qwen3-vl:8b
MLX_MODEL ?= mlx-community/Qwen3-VL-8B-Instruct-4bit
CORPUS ?= eval/corpus

.PHONY: help setup up up-gpu down logs models serve-llm-ollama serve-llm-mlx ingest query serve ui eval test test-integration lint fmt typecheck check

help:
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: ## Install all dependencies and git hooks
	uv sync --all-extras --group dev
	[ -f .env ] || cp .env.example .env
	uv run pre-commit install

up: ## Start Qdrant and Langfuse
	docker compose up -d

up-gpu: ## Start Qdrant, Langfuse and vLLM (NVIDIA GPU host)
	docker compose --profile gpu up -d

down: ## Stop all services
	docker compose --profile gpu --profile app down

logs: ## Tail service logs
	docker compose logs -f --tail=100

models: ## Download all models from configs/models.yaml
	uv run rag models pull-all

serve-llm-ollama: ## Pull and serve the generator with Ollama
	ollama pull $(OLLAMA_MODEL)
	ollama serve

serve-llm-mlx: ## Serve the generator with mlx-vlm (Apple Silicon)
	uv run --extra mlx python -m mlx_vlm.server --model $(MLX_MODEL) --host 127.0.0.1 --port 8080

ingest: ## Ingest the sample corpus (override with CORPUS=path)
	uv run rag ingest $(CORPUS)

query: ## Ask a question: make query Q="..."
	uv run rag query "$(Q)"

serve: ## Run the FastAPI server
	uv run rag serve

ui: ## Run the Gradio chat UI
	uv run rag ui

eval: ## Run the RAGAS evaluation with threshold gate
	uv run rag eval

test: ## Run unit tests
	uv run pytest

test-integration: ## Run integration tests (requires Docker)
	uv run pytest -m integration

lint: ## Lint and check formatting
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Format code
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## Static type checks
	uv run pyright

check: lint typecheck test ## Lint, type check and test
