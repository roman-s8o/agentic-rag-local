from __future__ import annotations

import logging
from collections.abc import Sequence

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from agentic_rag.config import LLMSettings
from agentic_rag.providers.base import (
    LLMProvider,
    StructuredOutputError,
    T,
    json_instruction,
    message_text,
    parse_model,
)

logger = logging.getLogger(__name__)

JSON_SCHEMA_BACKENDS = frozenset({"vllm", "ollama", "openai"})


class OpenAICompatibleProvider(LLMProvider):
    """Any server exposing /v1/chat/completions: vLLM, Ollama, mlx-vlm, OpenAI."""

    def __init__(self, settings: LLMSettings, max_parse_retries: int = 1) -> None:
        self.settings = settings
        self.max_parse_retries = max_parse_retries
        mode = settings.structured_output
        self.use_json_schema = mode == "json_schema" or (
            mode == "auto" and settings.backend in JSON_SCHEMA_BACKENDS
        )
        self._chat_model: BaseChatModel | None = None

    @property
    def chat_model(self) -> BaseChatModel:
        if self._chat_model is None:
            s = self.settings
            self._chat_model = ChatOpenAI(
                model=s.model,
                base_url=s.resolved_base_url,
                api_key=s.api_key,
                temperature=s.temperature,
                max_completion_tokens=s.max_tokens,
                timeout=s.timeout_s,
                max_retries=2,
                reasoning_effort=s.reasoning_effort,
            )
        return self._chat_model

    def structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        if self.use_json_schema:
            try:
                runnable = self.chat_model.with_structured_output(schema, method="json_schema")
                result = runnable.invoke(list(messages))
                if isinstance(result, schema):
                    return result
                return schema.model_validate(result)
            except Exception as exc:
                logger.warning("json_schema structured output failed, using prompt mode: %s", exc)
        return self._structured_via_prompt(messages, schema)

    def _structured_via_prompt(self, messages: Sequence[BaseMessage], schema: type[T]) -> T:
        convo = [*messages, json_instruction(schema)]
        last_error: StructuredOutputError | None = None
        for _ in range(self.max_parse_retries + 1):
            text = message_text(self.chat_model.invoke(convo))
            try:
                return parse_model(text, schema)
            except StructuredOutputError as exc:
                last_error = exc
                convo = [
                    *convo,
                    HumanMessage(content=f"Invalid JSON ({exc}). Return only valid JSON."),
                ]
        raise StructuredOutputError(f"Could not parse {schema.__name__}: {last_error}")
