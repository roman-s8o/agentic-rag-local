from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage


def to_data_uri(image: str | Path) -> str:
    if isinstance(image, str) and image.startswith(("data:", "http://", "https://")):
        return image
    path = Path(image)
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"


def image_part(image: str | Path) -> dict[str, Any]:
    return {"type": "image_url", "image_url": {"url": to_data_uri(image)}}


def user_message(text: str, images: list[str | Path] | None = None) -> HumanMessage:
    if not images:
        return HumanMessage(content=text)
    parts: list[str | dict[str, Any]] = [image_part(i) for i in images]
    parts.append({"type": "text", "text": text})
    return HumanMessage(content=parts)
