from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from pydantic import BaseModel, Field, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

Backend = Literal["ollama", "mlx", "vllm", "openai"]

DEFAULT_BASE_URLS: dict[str, str] = {
    "ollama": "http://localhost:11434/v1",
    "mlx": "http://localhost:8080/v1",
    "vllm": "http://localhost:8000/v1",
    "openai": "https://api.openai.com/v1",
}

CONFIG_DIR = Path(os.environ.get("RAG_CONFIG_DIR", "configs"))


class LLMSettings(BaseModel):
    backend: Backend = "vllm"
    base_url: str | None = None
    model: str = "Qwen/Qwen3-VL-8B-Instruct"
    api_key: SecretStr = SecretStr("local")
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout_s: float = 180.0
    structured_output: Literal["auto", "json_schema", "prompt"] = "auto"
    reasoning_effort: str | None = None

    @property
    def resolved_base_url(self) -> str:
        return self.base_url or DEFAULT_BASE_URLS[self.backend]


class EmbeddingSettings(BaseModel):
    text_backend: Literal["bgem3", "sentence_transformers"] = "bgem3"
    dense_model: str = "BAAI/bge-m3"
    sparse_backend: Literal["bgem3", "bm25"] = "bgem3"
    sparse_model: str = "Qdrant/bm25"
    image_enabled: bool = True
    image_model: str = "vidore/colqwen2.5-v0.2"
    device: str | None = None
    batch_size: int = 16


class RerankerSettings(BaseModel):
    backend: Literal["cross_encoder", "http", "none"] = "cross_encoder"
    model: str = "BAAI/bge-reranker-v2-m3"
    base_url: str | None = None
    top_n: int = 6


class QdrantSettings(BaseModel):
    url: str | None = "http://localhost:6333"
    path: Path | None = None
    api_key: SecretStr | None = None
    collection: str = "documents"


class RetrievalSettings(BaseModel):
    prefetch_limit: int = 30
    limit: int = 20
    rrf_k: int = 60


class IngestionSettings(BaseModel):
    max_chunk_tokens: int = 512
    caption_images: bool = True
    min_image_px: int = 96
    images_dir: Path = Path("data/images")
    uploads_dir: Path = Path("data/uploads")


class AgentSettings(BaseModel):
    max_retrieval_rounds: int = 2
    max_generation_attempts: int = 2
    max_sub_queries: int = 3
    max_context_images: int = 3
    context_chars_per_doc: int = 1500
    checkpoint_path: Path = Path("data/checkpoints.sqlite")


class EvalSettings(BaseModel):
    dataset: Path = Path("eval/datasets/golden.jsonl")
    output_dir: Path = Path("eval/results")
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "faithfulness": 0.75,
            "answer_relevancy": 0.70,
            "context_precision": 0.65,
            "context_recall": 0.70,
        }
    )
    judge: LLMSettings | None = None


class ObservabilitySettings(BaseModel):
    langfuse_enabled: bool = True
    tags: list[str] = Field(default_factory=lambda: ["agentic-rag"])


class APISettings(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8800


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_nested_delimiter="__",
        env_file=".env",
        extra="ignore",
        yaml_file=CONFIG_DIR / "settings.yaml",
    )

    models_dir: Path = Path("models")
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    ingestion: IngestionSettings = Field(default_factory=IngestionSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    eval: EvalSettings = Field(default_factory=EvalSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    api: APISettings = Field(default_factory=APISettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    load_dotenv(override=False)
    settings = Settings()
    os.environ.setdefault("HF_HUB_CACHE", str(settings.models_dir.resolve()))
    return settings
