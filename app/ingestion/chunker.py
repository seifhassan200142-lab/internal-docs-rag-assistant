from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from app.ingestion.loaders import LoadedDocument, LoadedPage
from app.schemas.models import ChunkMetadata, DocumentChunk


def split_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> list[str]:
    """Split text into overlapping, character-based chunks while respecting word boundaries."""
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    text_length = len(cleaned)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        if end < text_length:
            whitespace = cleaned.rfind(" ", start, end)
            if whitespace > start + int(chunk_size * 0.5):
                end = whitespace

        chunk = cleaned[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break
        start = max(end - chunk_overlap, 0)

    return chunks


def chunk_loaded_document(
    document: LoadedDocument,
    document_id: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[DocumentChunk]:
    created_at = datetime.now(timezone.utc)
    chunks: list[DocumentChunk] = []

    for page in document.pages:
        for text_chunk in split_text(page.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap):
            chunk_number = len(chunks) + 1
            chunk_id = f"{document_id}_{chunk_number:04d}"
            chunks.append(
                DocumentChunk(
                    text=text_chunk,
                    metadata=ChunkMetadata(
                        document_id=document_id,
                        document_name=document.document_name,
                        chunk_id=chunk_id,
                        page_number=page.page_number,
                        source_file_path=document.source_file_path,
                        created_at=created_at,
                    ),
                )
            )

    return chunks
