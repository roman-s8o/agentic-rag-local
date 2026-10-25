from agentic_rag.config import LLMSettings
from agentic_rag.providers.base import LLMProvider, StructuredOutputError
from agentic_rag.providers.multimodal import to_data_uri, user_message
from agentic_rag.providers.openai_compat import OpenAICompatibleProvider


def build_llm(settings: LLMSettings) -> LLMProvider:
    return OpenAICompatibleProvider(settings)


__all__ = [
    "LLMProvider",
    "OpenAICompatibleProvider",
    "StructuredOutputError",
    "build_llm",
    "to_data_uri",
    "user_message",
]
