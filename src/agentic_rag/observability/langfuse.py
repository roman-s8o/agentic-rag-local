from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig

from agentic_rag.config import ObservabilitySettings

logger = logging.getLogger(__name__)


def langfuse_configured(settings: ObservabilitySettings) -> bool:
    return settings.langfuse_enabled and bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


@lru_cache
def _client() -> Any:
    from langfuse import get_client

    return get_client()


def new_handler(settings: ObservabilitySettings) -> BaseCallbackHandler | None:
    if not langfuse_configured(settings):
        return None
    try:
        from langfuse.langchain import CallbackHandler

        _client()
        return CallbackHandler()
    except Exception as exc:
        logger.warning("Langfuse disabled: %s", exc)
        return None


def run_config(
    settings: ObservabilitySettings,
    thread_id: str,
    handler: BaseCallbackHandler | None = None,
    user_id: str | None = None,
    tags: Sequence[str] = (),
    recursion_limit: int = 40,
) -> RunnableConfig:
    metadata: dict[str, Any] = {
        "langfuse_session_id": thread_id,
        "langfuse_tags": [*settings.tags, *tags],
    }
    if user_id:
        metadata["langfuse_user_id"] = user_id
    return RunnableConfig(
        configurable={"thread_id": thread_id},
        callbacks=[handler] if handler else [],
        metadata=metadata,
        run_name="agentic-rag",
        recursion_limit=recursion_limit,
    )


def trace_id(handler: BaseCallbackHandler | None) -> str | None:
    return getattr(handler, "last_trace_id", None) if handler else None


def score(trace: str | None, name: str, value: float, comment: str | None = None) -> None:
    if not trace:
        return
    try:
        _client().create_score(trace_id=trace, name=name, value=value, comment=comment)
    except Exception as exc:
        logger.warning("Could not push score %s to Langfuse: %s", name, exc)


def flush(settings: ObservabilitySettings) -> None:
    if langfuse_configured(settings):
        _client().flush()
