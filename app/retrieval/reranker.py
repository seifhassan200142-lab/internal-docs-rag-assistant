from __future__ import annotations

from app.schemas.models import RetrievedChunk


class CrossEncoderReranker:
    def __init__(self, model_name: str, enabled: bool):
        self.model_name = model_name
        self.enabled = enabled
        self._model = None

    @property
    def model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as exc:
                raise RuntimeError("sentence-transformers is required for reranking. Install requirements.txt first.") from exc
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, question: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not self.enabled or not chunks:
            return chunks[:top_k]

        pairs = [(question, chunk.text) for chunk in chunks]
        scores = self.model.predict(pairs)
        for chunk, score in zip(chunks, scores):
            chunk.rerank_score = float(score)

        return sorted(chunks, key=lambda item: item.rerank_score or 0.0, reverse=True)[:top_k]
