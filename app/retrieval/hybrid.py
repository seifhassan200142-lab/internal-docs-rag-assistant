from __future__ import annotations

from app.schemas.models import RetrievedChunk


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievedChunk]],
    k: int = 60,
    top_k: int = 6,
) -> list[RetrievedChunk]:
    """Combine multiple ranked result lists using Reciprocal Rank Fusion."""
    by_chunk_id: dict[str, RetrievedChunk] = {}
    fusion_scores: dict[str, float] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            by_chunk_id.setdefault(chunk.chunk_id, chunk)
            fusion_scores[chunk.chunk_id] = fusion_scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)

            existing = by_chunk_id[chunk.chunk_id]
            if chunk.dense_score is not None:
                existing.dense_score = chunk.dense_score
            if chunk.sparse_score is not None:
                existing.sparse_score = chunk.sparse_score

    fused = []
    for chunk_id, score in fusion_scores.items():
        chunk = by_chunk_id[chunk_id]
        chunk.fusion_score = score
        fused.append(chunk)

    return sorted(fused, key=lambda item: item.fusion_score or 0.0, reverse=True)[:top_k]


class HybridRetriever:
    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def fuse(
        self,
        dense_results: list[RetrievedChunk],
        sparse_results: list[RetrievedChunk],
        top_k: int,
    ) -> list[RetrievedChunk]:
        return reciprocal_rank_fusion(
            [dense_results, sparse_results],
            k=self.rrf_k,
            top_k=top_k,
        )
