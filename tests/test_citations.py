from app.schemas.models import RetrievedChunk
from app.utils.citations import build_citations, format_citation, make_preview


def test_make_preview_truncates_long_text():
    preview = make_preview("word " * 100, max_length=30)

    assert len(preview) <= 30
    assert preview.endswith("...")


def test_format_citation_includes_required_fields():
    chunk = RetrievedChunk(
        document_id="doc-1",
        document_name="handbook.md",
        chunk_id="doc-1_0001",
        text="Remote work is allowed with manager approval.",
        page_number=4,
        source_file_path="handbook.md",
    )

    citation = format_citation(chunk)

    assert citation.document_name == "handbook.md"
    assert citation.page_number == 4
    assert citation.chunk_id == "doc-1_0001"
    assert "Remote work" in citation.preview


def test_build_citations_deduplicates_chunks():
    chunk = RetrievedChunk(
        document_id="doc-1",
        document_name="handbook.md",
        chunk_id="doc-1_0001",
        text="Text",
        source_file_path="handbook.md",
    )

    citations = build_citations([chunk, chunk])

    assert len(citations) == 1
