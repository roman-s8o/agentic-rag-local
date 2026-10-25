from agentic_rag.config import LLMSettings, Settings, get_settings


def test_yaml_defaults_load() -> None:
    get_settings.cache_clear()
    settings = get_settings()
    assert settings.qdrant.collection == "documents"
    assert settings.retrieval.prefetch_limit == 30
    assert settings.eval.thresholds["faithfulness"] == 0.75
    get_settings.cache_clear()


def test_env_overrides_yaml(monkeypatch) -> None:
    monkeypatch.setenv("RAG_LLM__BACKEND", "vllm")
    monkeypatch.setenv("RAG_LLM__MODEL", "Qwen/Qwen3-VL-8B-Instruct")
    settings = Settings()
    assert settings.llm.backend == "vllm"
    assert settings.llm.model == "Qwen/Qwen3-VL-8B-Instruct"
    assert settings.llm.resolved_base_url == "http://localhost:8000/v1"


def test_backend_urls() -> None:
    assert LLMSettings(backend="ollama").resolved_base_url.endswith(":11434/v1")
    assert LLMSettings(backend="mlx").resolved_base_url.endswith(":8080/v1")
