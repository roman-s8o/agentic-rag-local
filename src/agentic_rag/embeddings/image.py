from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from agentic_rag.embeddings.device import resolve_device

if TYPE_CHECKING:
    from PIL.Image import Image

MultiVector = list[list[float]]


class ImageEmbedder(Protocol):
    dim: int

    def embed_images(self, images: list[Image]) -> list[MultiVector]: ...

    def embed_query(self, text: str) -> MultiVector: ...


class ColQwenEmbedder:
    """Late-interaction (ColPali-style) multivector embeddings for images and text queries."""

    def __init__(
        self,
        model: str = "vidore/colqwen2.5-v0.2",
        device: str | None = None,
        batch_size: int = 4,
    ) -> None:
        import torch
        from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor

        self._torch = torch
        dev = resolve_device(device)
        dtype = torch.bfloat16 if dev in ("cuda", "mps") else torch.float32
        self._model = ColQwen2_5.from_pretrained(model, torch_dtype=dtype, device_map=dev).eval()
        self._processor = ColQwen2_5_Processor.from_pretrained(model)
        self.batch_size = batch_size
        self.dim = 128

    def _run(self, batch: Any) -> list[MultiVector]:
        batch = {k: v.to(self._model.device) for k, v in batch.items()}
        with self._torch.no_grad():
            emb = self._model(**batch)
        mask = batch["attention_mask"].bool()
        return [e[m].float().cpu().tolist() for e, m in zip(emb, mask, strict=True)]

    def embed_images(self, images: list[Image]) -> list[MultiVector]:
        out: list[MultiVector] = []
        for i in range(0, len(images), self.batch_size):
            rgb = [img.convert("RGB") for img in images[i : i + self.batch_size]]
            out.extend(self._run(self._processor.process_images(rgb)))
        return out

    def embed_query(self, text: str) -> MultiVector:
        return self._run(self._processor.process_queries([text]))[0]
