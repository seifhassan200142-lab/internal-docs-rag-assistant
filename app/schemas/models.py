from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    page_number: Optional[int] = None
    source_file_path: str
    created_at: datetime


class DocumentChunk(BaseModel):
    text: str
    metadata: ChunkMetadata


class DocumentInfo(BaseModel):
    document_id: str
    document_name: str
    source_file_path: str
    chunk_count: int
    created_at: datetime


class IngestResponse(BaseModel):
    document_id: str
    document_name: str
    chunks_indexed: int
    message: str


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=6, ge=1, le=30)
    include_chunks: bool = True


class Citation(BaseModel):
    document_name: str
    page_number: Optional[int] = None
    chunk_id: str
    preview: str


class RetrievedChunk(BaseModel):
    document_id: str
    document_name: str
    chunk_id: str
    text: str
    page_number: Optional[int] = None
    source_file_path: str
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    used_reranking: bool = False
    top_k: int


class HealthResponse(BaseModel):
    status: str
    service: str
