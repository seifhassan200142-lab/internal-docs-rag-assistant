from __future__ import annotations

import hashlib
from typing import Any

from app.schemas.models import DocumentChunk, RetrievedChunk


class EmbeddingService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for embeddings. Install requirements.txt first.") from exc
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_texts([query])[0]


class QdrantVectorStore:
    def __init__(self, url: str, collection_name: str):
        self.url = url
        self.collection_name = collection_name
        self._client = None

    @property
    def client(self):
        if self._client is None:
            try:
                from qdrant_client import QdrantClient
            except ImportError as exc:
                raise RuntimeError("qdrant-client is required. Install requirements.txt first.") from exc
            self._client = QdrantClient(url=self.url)
        return self._client

    def ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.http import models as qmodels

        collections = self.client.get_collections().collections
        if any(collection.name == self.collection_name for collection in collections):
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=qmodels.VectorParams(size=vector_size, distance=qmodels.Distance.COSINE),
        )

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        from qdrant_client.http import models as qmodels

        if len(chunks) != len(embeddings):
            raise ValueError("Number of chunks must match number of embeddings.")

        points = []
        for chunk, embedding in zip(chunks, embeddings):
            metadata = chunk.metadata.model_dump(mode="json")
            payload: dict[str, Any] = {**metadata, "text": chunk.text}
            points.append(
                qmodels.PointStruct(
                    id=_stable_uuid_from_chunk_id(chunk.metadata.chunk_id),
                    vector=embedding,
                    payload=payload,
                )
            )

        self.client.upsert(collection_name=self.collection_name, points=points)

    def search(self, query_embedding: list[float], top_k: int) -> list[RetrievedChunk]:
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for point in results:
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    document_id=str(payload.get("document_id", "")),
                    document_name=str(payload.get("document_name", "")),
                    chunk_id=str(payload.get("chunk_id", "")),
                    text=str(payload.get("text", "")),
                    page_number=payload.get("page_number"),
                    source_file_path=str(payload.get("source_file_path", "")),
                    dense_score=float(point.score),
                    metadata=payload,
                )
            )
        return chunks

    def delete_document(self, document_id: str) -> None:
        from qdrant_client.http import models as qmodels

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )


def _stable_uuid_from_chunk_id(chunk_id: str) -> str:
    digest = hashlib.md5(chunk_id.encode("utf-8"), usedforsecurity=False).hexdigest()
    return f"{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"
