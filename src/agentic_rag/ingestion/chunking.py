from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class TextPiece:
    text: str
    page: int | None = None
    headings: list[str] = field(default_factory=list)


def split_text(text: str, max_chars: int = 2000, overlap: int = 200) -> list[str]:
    """Paragraph-aware splitter with character overlap, used for plain-text files."""
    if overlap >= max_chars:
        raise ValueError("overlap must be smaller than max_chars")
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        while len(para) > max_chars:
            head, para = para[:max_chars], para[max_chars - overlap :]
            if current:
                chunks.append(current)
                current = ""
            chunks.append(head)
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= max_chars:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            with_tail = f"{tail}\n\n{para}"
            current = with_tail if tail and len(with_tail) <= max_chars else para
    if current:
        chunks.append(current)
    return chunks


def chunk_docling(doc: Any, tokenizer_model: str, max_tokens: int) -> list[TextPiece]:
    """Structure-aware, token-bounded chunking with heading context (Docling HybridChunker)."""
    from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
    from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
    from transformers import AutoTokenizer

    tokenizer = HuggingFaceTokenizer(
        tokenizer=AutoTokenizer.from_pretrained(tokenizer_model), max_tokens=max_tokens
    )
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    pieces: list[TextPiece] = []
    for chunk in chunker.chunk(dl_doc=doc):
        text = chunker.contextualize(chunk=chunk).strip()
        if not text:
            continue
        meta: Any = chunk.meta
        pages = sorted({p.page_no for item in meta.doc_items for p in item.prov})
        pieces.append(
            TextPiece(
                text=text,
                page=pages[0] if pages else None,
                headings=list(meta.headings or []),
            )
        )
    return pieces
