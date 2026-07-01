from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import get_settings
from app.core.document_store import DocumentStore
from app.generation.llm_client import LLMClient, LLMConfigurationError
from app.generation.prompts import INSUFFICIENT_CONTEXT_MESSAGE
from app.ingestion.pipeline import IngestionPipeline
from app.retrieval.dense import EmbeddingService, QdrantVectorStore
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.sparse import SparseBM25Retriever
from app.schemas.models import DocumentInfo, HealthResponse, IngestResponse, QueryRequest, QueryResponse
from app.utils.citations import build_citations

router = APIRouter()
settings = get_settings()
document_store = DocumentStore(settings.index_dir)
embedding_service = EmbeddingService(settings.embedding_model)
vector_store = QdrantVectorStore(settings.qdrant_url, settings.qdrant_collection_name)
hybrid_retriever = HybridRetriever(rrf_k=settings.rrf_k)
reranker = CrossEncoderReranker(model_name=settings.reranker_model, enabled=settings.enable_reranking)
llm_client = LLMClient(settings)


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name)


@router.post("/api/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)) -> IngestResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a filename.")

    suffix = Path(file.filename).suffix
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(await file.read())

        pipeline = IngestionPipeline(
            settings=settings,
            embedding_service=embedding_service,
            vector_store=vector_store,
            document_store=document_store,
        )
        response = pipeline.ingest_file(temp_path)
        return response
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Document ingestion failed: {exc}") from exc
    finally:
        if "temp_path" in locals() and temp_path.exists():
            temp_path.unlink(missing_ok=True)


@router.post("/api/query", response_model=QueryResponse)
def query_documents(request: QueryRequest) -> QueryResponse:
    chunks = document_store.list_chunks()
    if not chunks:
        return QueryResponse(
            question=request.question,
            answer=INSUFFICIENT_CONTEXT_MESSAGE,
            citations=[],
            retrieved_chunks=[],
            used_reranking=settings.enable_reranking,
            top_k=request.top_k,
        )

    try:
        query_embedding = embedding_service.embed_query(request.question)
        dense_results = vector_store.search(query_embedding=query_embedding, top_k=request.top_k * 2)
        sparse_results = SparseBM25Retriever(chunks).search(request.question, top_k=request.top_k * 2)
        fused_results = hybrid_retriever.fuse(dense_results, sparse_results, top_k=request.top_k * 2)
        ranked_results = reranker.rerank(request.question, fused_results, top_k=request.top_k)

        if len(ranked_results) < settings.min_context_chunks:
            answer = INSUFFICIENT_CONTEXT_MESSAGE
            citations = []
        else:
            answer = llm_client.generate_answer(request.question, ranked_results)
            citations = [] if answer.strip() == INSUFFICIENT_CONTEXT_MESSAGE else build_citations(ranked_results)

        return QueryResponse(
            question=request.question,
            answer=answer,
            citations=citations,
            retrieved_chunks=ranked_results if request.include_chunks else [],
            used_reranking=settings.enable_reranking,
            top_k=request.top_k,
        )
    except LLMConfigurationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Query failed: {exc}") from exc


@router.get("/api/documents", response_model=list[DocumentInfo])
def list_documents() -> list[DocumentInfo]:
    return document_store.list_documents()


@router.delete("/api/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, str | int]:
    if not any(document.document_id == document_id for document in document_store.list_documents()):
        raise HTTPException(status_code=404, detail="Document not found.")

    try:
        vector_store.delete_document(document_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Qdrant deletion failed: {exc}") from exc

    removed_chunks = document_store.remove_document(document_id)
    return {"document_id": document_id, "removed_chunks": removed_chunks, "message": "Document deleted."}
