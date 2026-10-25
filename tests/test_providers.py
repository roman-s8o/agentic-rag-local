from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from agentic_rag.config import LLMSettings
from agentic_rag.providers import OpenAICompatibleProvider, user_message
from agentic_rag.providers.base import StructuredOutputError, extract_json, strip_reasoning


class Verdict(BaseModel):
    grounded: bool
    reason: str


def test_extract_json_handles_fences_and_reasoning() -> None:
    text = '<think>hmm {"x": 0}</think>Here:\n```json\n{"grounded": true, "reason": "ok"}\n```'
    assert extract_json(text) == {"grounded": True, "reason": "ok"}
    assert strip_reasoning("<think>a</think> b") == "b"


def test_extract_json_raises_without_json() -> None:
    with pytest.raises(StructuredOutputError):
        extract_json("no json here")


def _provider(responses: list[str]) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider(LLMSettings(backend="mlx", model="fake"))
    provider._chat_model = FakeListChatModel(responses=responses)
    return provider


def test_prompt_mode_retries_on_invalid_json() -> None:
    provider = _provider(["not json", '{"grounded": false, "reason": "missing"}'])
    assert not provider.use_json_schema
    result = provider.structured([HumanMessage(content="q")], Verdict)
    assert result == Verdict(grounded=False, reason="missing")


def test_prompt_mode_gives_up() -> None:
    provider = _provider(["nope", "still nope"])
    with pytest.raises(StructuredOutputError):
        provider.structured([HumanMessage(content="q")], Verdict)


def test_backend_capabilities() -> None:
    assert OpenAICompatibleProvider(LLMSettings(backend="vllm")).use_json_schema
    assert not OpenAICompatibleProvider(LLMSettings(backend="mlx")).use_json_schema
    settings = LLMSettings(backend="vllm", structured_output="prompt")
    assert not OpenAICompatibleProvider(settings).use_json_schema
    assert LLMSettings(backend="vllm").resolved_base_url == "http://localhost:8000/v1"


def test_user_message_embeds_images(tmp_path: Path) -> None:
    img = tmp_path / "a.png"
    img.write_bytes(b"\x89PNG\r\n")
    msg = user_message("describe", [img])
    assert isinstance(msg.content, list)
    first = msg.content[0]
    assert isinstance(first, dict)
    assert first["image_url"]["url"].startswith("data:image/png;base64,")
