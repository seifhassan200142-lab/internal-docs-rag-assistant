from __future__ import annotations

import re

from app.schemas.models import Citation, RetrievedChunk


def make_preview(text: str, max_length: int = 180) -> str:
    preview = re.sub(r"\s+", " ", text).strip()
    if len(preview) <= max_length:
        return preview
    return preview[: max_length - 3].rstrip() + "..."


def format_citation(chunk: RetrievedChunk, max_preview_length: int = 180) -> Citation:
    return Citation(
        document_name=chunk.document_name,
        page_number=chunk.page_number,
        chunk_id=chunk.chunk_id,
        preview=make_preview(chunk.text, max_length=max_preview_length),
    )


def build_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    seen: set[str] = set()
    citations: list[Citation] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        citations.append(format_citation(chunk))
    return citations
