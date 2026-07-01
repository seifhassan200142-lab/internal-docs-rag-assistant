from datetime import datetime, timezone

from app.retrieval.sparse import SparseBM25Retriever
from app.schemas.models import ChunkMetadata, DocumentChunk


def _chunk(chunk_id: str, text: str) -> DocumentChunk:
    return DocumentChunk(
        text=text,
        metadata=ChunkMetadata(
            document_id="doc-1",
            document_name="sample.md",
            chunk_id=chunk_id,
            page_number=None,
            source_file_path="sample.md",
            created_at=datetime.now(timezone.utc),
        ),
    )


def test_bm25_retrieves_keyword_matching_chunk_first():
    chunks = [
        _chunk("c1", "The remote work policy allows employees to work from home."),
        _chunk("c2", "Travel expenses require receipts after business trips."),
        _chunk("c3", "The office kitchen is cleaned every Friday."),
    ]

    results = SparseBM25Retriever(chunks).search("remote work policy", top_k=2)

    assert results
    assert results[0].chunk_id == "c1"
    assert results[0].sparse_score is not None
