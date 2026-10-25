from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agentic_rag.ingestion.chunking import TextPiece, chunk_docling

if TYPE_CHECKING:
    from PIL.Image import Image

DOCLING_EXTENSIONS = frozenset(
    {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm", ".md", ".adoc", ".csv"}
)


@dataclass
class ParsedPicture:
    image: Image
    page: int | None = None
    caption: str = ""


@dataclass
class ParsedDocument:
    texts: list[TextPiece] = field(default_factory=list)
    pictures: list[ParsedPicture] = field(default_factory=list)


class DoclingParser:
    def __init__(self, tokenizer_model: str, max_tokens: int = 512, images_scale: float = 2.0):
        self.tokenizer_model = tokenizer_model
        self.max_tokens = max_tokens
        self.images_scale = images_scale

    @cached_property
    def converter(self) -> Any:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption

        options = PdfPipelineOptions()
        options.generate_picture_images = True
        options.images_scale = self.images_scale
        return DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )

    def parse(self, path: Path) -> ParsedDocument:
        doc = self.converter.convert(str(path)).document
        parsed = ParsedDocument(texts=chunk_docling(doc, self.tokenizer_model, self.max_tokens))
        for picture in doc.pictures:
            image = picture.get_image(doc)
            if image is None:
                continue
            parsed.pictures.append(
                ParsedPicture(
                    image=image,
                    page=picture.prov[0].page_no if picture.prov else None,
                    caption=picture.caption_text(doc) or "",
                )
            )
        return parsed
