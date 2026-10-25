from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from typing import Any, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class StructuredOutputError(RuntimeError):
    pass


def message_text(message: Any) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def strip_reasoning(text: str) -> str:
    return _THINK_RE.sub("", text).strip()


def extract_json(text: str) -> Any:
    text = strip_reasoning(text)
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
    if start == -1:
        raise StructuredOutputError(f"No JSON found in model output: {text[:200]!r}")
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(text[start:])
    return obj


def parse_model(text: str, schema: type[T]) -> T:
    try:
        return schema.model_validate(extract_json(text))
    except (ValidationError, json.JSONDecodeError) as exc:
        raise StructuredOutputError(str(exc)) from exc


def json_instruction(schema: type[BaseModel]) -> SystemMessage:
    return SystemMessage(
        content=(
            "Respond with a single JSON object that validates against this JSON schema. "
            "Output JSON only, no prose.\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
    )


class LLMProvider(ABC):
    @property
    @abstractmethod
    def chat_model(self) -> BaseChatModel: ...

    @abstractmethod
    def structured(self, messages: Sequence[BaseMessage], schema: type[T]) -> T: ...

    def complete(self, messages: Sequence[BaseMessage]) -> str:
        return strip_reasoning(message_text(self.chat_model.invoke(list(messages))))

    def stream(self, messages: Sequence[BaseMessage]) -> Iterator[str]:
        for chunk in self.chat_model.stream(list(messages)):
            text = message_text(chunk)
            if text:
                yield text
