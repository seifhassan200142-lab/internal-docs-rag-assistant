from app.retrieval.hybrid import reciprocal_rank_fusion
from app.schemas.models import RetrievedChunk


def _result(chunk_id: str, dense_score=None, sparse_score=None):
    return RetrievedChunk(
        document_id="doc-1",
        document_name="sample.md",
        chunk_id=chunk_id,
        text=f"Text for {chunk_id}",
        source_file_path="sample.md",
        dense_score=dense_score,
        sparse_score=sparse_score,
    )


def test_reciprocal_rank_fusion_merges_and_scores_results():
    dense = [_result("a", dense_score=0.9), _result("b", dense_score=0.8)]
    sparse = [_result("b", sparse_score=3.0), _result("c", sparse_score=2.0)]

    fused = reciprocal_rank_fusion([dense, sparse], k=60, top_k=3)

    assert [item.chunk_id for item in fused][:1] == ["b"]
    assert len(fused) == 3
    assert fused[0].fusion_score is not None
    assert fused[0].dense_score is not None
    assert fused[0].sparse_score is not None
