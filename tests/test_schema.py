from app.schemas.models import Citation, QueryResponse, RetrievedChunk


def test_query_response_schema():
    response = QueryResponse(
        question="What is the remote work policy?",
        answer="Employees may work remotely up to three days per week.",
        citations=[
            Citation(
                document_name="company_handbook.md",
                page_number=None,
                chunk_id="doc_0001",
                preview="Employees may work remotely up to three days per week.",
            )
        ],
        retrieved_chunks=[
            RetrievedChunk(
                document_id="doc",
                document_name="company_handbook.md",
                chunk_id="doc_0001",
                text="Employees may work remotely up to three days per week.",
                source_file_path="sample_data/company_handbook.md",
                fusion_score=0.032,
            )
        ],
        used_reranking=False,
        top_k=6,
    )

    data = response.model_dump()

    assert data["question"] == "What is the remote work policy?"
    assert data["citations"][0]["chunk_id"] == "doc_0001"
    assert data["retrieved_chunks"][0]["fusion_score"] == 0.032
