from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

from app.schemas.models import DocumentChunk, DocumentInfo


class DocumentStore:
    """Small local JSONL metadata store used by BM25 and document listing."""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks_path = self.index_dir / "chunks.jsonl"

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        with self.chunks_path.open("a", encoding="utf-8") as file:
            for chunk in chunks:
                file.write(chunk.model_dump_json() + "\n")

    def list_chunks(self) -> list[DocumentChunk]:
        if not self.chunks_path.exists():
            return []

        chunks: list[DocumentChunk] = []
        with self.chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                chunks.append(DocumentChunk.model_validate(json.loads(line)))
        return chunks

    def list_documents(self) -> list[DocumentInfo]:
        documents: OrderedDict[str, dict] = OrderedDict()
        for chunk in self.list_chunks():
            meta = chunk.metadata
            if meta.document_id not in documents:
                documents[meta.document_id] = {
                    "document_id": meta.document_id,
                    "document_name": meta.document_name,
                    "source_file_path": meta.source_file_path,
                    "chunk_count": 0,
                    "created_at": meta.created_at,
                }
            documents[meta.document_id]["chunk_count"] += 1
        return [DocumentInfo(**doc) for doc in documents.values()]

    def remove_document(self, document_id: str) -> int:
        chunks = self.list_chunks()
        kept = [chunk for chunk in chunks if chunk.metadata.document_id != document_id]
        removed = len(chunks) - len(kept)

        with self.chunks_path.open("w", encoding="utf-8") as file:
            for chunk in kept:
                file.write(chunk.model_dump_json() + "\n")
        return removed
