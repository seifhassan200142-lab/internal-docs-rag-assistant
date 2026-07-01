from __future__ import annotations

import shutil
import uuid
from pathlib import Path

from app.core.config import Settings
from app.core.document_store import DocumentStore
from app.ingestion.chunker import chunk_loaded_document
from app.ingestion.loaders import load_document
from app.retrieval.dense import EmbeddingService, QdrantVectorStore
from app.schemas.models import IngestResponse


class IngestionPipeline:
    def __init__(
        self,
        settings: Settings,
        embedding_service: EmbeddingService,
        vector_store: QdrantVectorStore,
        document_store: DocumentStore,
    ):
        self.settings = settings
        self.embedding_service = embedding_service
        self.vector_store = vector_store
        self.document_store = document_store

    def ingest_file(self, source_path: Path) -> IngestResponse:
        document_id = str(uuid.uuid4())
        stored_path = self._copy_to_upload_dir(source_path, document_id)
        loaded_document = load_document(stored_path)
        chunks = chunk_loaded_document(
            loaded_document,
            document_id=document_id,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
        )

        if not chunks:
            raise ValueError("No extractable text was found in this document.")

        embeddings = self.embedding_service.embed_texts([chunk.text for chunk in chunks])
        self.vector_store.ensure_collection(vector_size=len(embeddings[0]))
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        self.document_store.add_chunks(chunks)

        return IngestResponse(
            document_id=document_id,
            document_name=loaded_document.document_name,
            chunks_indexed=len(chunks),
            message="Document indexed successfully.",
        )

    def _copy_to_upload_dir(self, source_path: Path, document_id: str) -> Path:
        safe_name = source_path.name.replace("/", "_").replace("\\", "_")
        target_path = self.settings.upload_dir / f"{document_id}_{safe_name}"
        shutil.copyfile(source_path, target_path)
        return target_path
