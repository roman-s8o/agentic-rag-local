from __future__ import annotations

import io
from pathlib import Path

from langchain_core.messages import SystemMessage
from PIL.Image import Image

from agentic_rag.providers import LLMProvider, user_message
from agentic_rag.schema import content_hash

IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".gif"})

CAPTION_PROMPT = (
    "Describe this image for a search index. Transcribe all visible text, labels, numbers, "
    "axis titles and legend entries verbatim. For charts and tables, list every data point. "
    "For diagrams, describe components and their relationships. Be factual and concise."
)


def image_bytes(image: Image) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return buf.getvalue()


def is_informative(image: Image, min_px: int) -> bool:
    return min(image.size) >= min_px


def save_image(image: Image, images_dir: Path) -> tuple[Path, str]:
    data = image_bytes(image)
    digest = content_hash(data)
    images_dir.mkdir(parents=True, exist_ok=True)
    path = images_dir / f"{digest[:24]}.png"
    if not path.exists():
        path.write_bytes(data)
    return path, digest


class ImageCaptioner:
    def __init__(self, llm: LLMProvider) -> None:
        self.llm = llm

    def caption(self, image_path: Path, context: str = "") -> str:
        prompt = CAPTION_PROMPT + (f"\nDocument context: {context}" if context else "")
        return self.llm.complete(
            [
                SystemMessage(content="You are a precise visual analyst."),
                user_message(prompt, [image_path]),
            ]
        )
