from app.ingestion.chunker import chunk_loaded_document, split_text
from app.ingestion.loaders import LoadedDocument, LoadedPage


def test_split_text_creates_overlapping_chunks():
    text = " ".join(["alpha beta gamma"] * 120)
    chunks = split_text(text, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert chunks[0] != chunks[1]


def test_chunk_loaded_document_preserves_metadata():
    document = LoadedDocument(
        document_name="policy.md",
        source_file_path="sample_data/policy.md",
        pages=[LoadedPage(text="Remote work policy text " * 40, page_number=2)],
    )

    chunks = chunk_loaded_document(document, document_id="doc-123", chunk_size=150, chunk_overlap=25)

    assert chunks
    assert chunks[0].metadata.document_id == "doc-123"
    assert chunks[0].metadata.document_name == "policy.md"
    assert chunks[0].metadata.page_number == 2
    assert chunks[0].metadata.chunk_id.startswith("doc-123_")
