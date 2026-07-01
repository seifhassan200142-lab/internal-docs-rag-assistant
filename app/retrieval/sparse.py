from __future__ import annotations

import math
import re
from collections import Counter

from app.schemas.models import DocumentChunk, RetrievedChunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_PATTERN.findall(text)]


class SparseBM25Retriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self._rank_bm25 = None
        self._fallback_index = None
        self._initialize()

    def _initialize(self) -> None:
        try:
            from rank_bm25 import BM25Okapi

            self._rank_bm25 = BM25Okapi(self.tokenized_corpus)
        except Exception:
            self._fallback_index = _FallbackBM25(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        if not self.chunks:
            return []

        query_tokens = tokenize(query)
        if self._rank_bm25 is not None:
            scores = self._rank_bm25.get_scores(query_tokens)
        else:
            scores = self._fallback_index.get_scores(query_tokens) if self._fallback_index else []

        ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
        results: list[RetrievedChunk] = []
        for index in ranked_indices[:top_k]:
            score = float(scores[index])
            if score <= 0:
                continue
            chunk = self.chunks[index]
            meta = chunk.metadata
            results.append(
                RetrievedChunk(
                    document_id=meta.document_id,
                    document_name=meta.document_name,
                    chunk_id=meta.chunk_id,
                    text=chunk.text,
                    page_number=meta.page_number,
                    source_file_path=meta.source_file_path,
                    sparse_score=score,
                    metadata=meta.model_dump(mode="json"),
                )
            )
        return results


class _FallbackBM25:
    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avgdl = sum(self.doc_lengths) / max(len(self.doc_lengths), 1)
        self.doc_freqs: Counter[str] = Counter()
        self.term_freqs = [Counter(doc) for doc in tokenized_corpus]
        for doc in tokenized_corpus:
            for term in set(doc):
                self.doc_freqs[term] += 1
        self.n_docs = len(tokenized_corpus)

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores: list[float] = []
        for doc_index, term_freq in enumerate(self.term_freqs):
            doc_len = self.doc_lengths[doc_index]
            score = 0.0
            for term in query_tokens:
                if term not in term_freq:
                    continue
                df = self.doc_freqs.get(term, 0)
                idf = math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))
                tf = term_freq[term]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1e-9))
                score += idf * numerator / denominator
            scores.append(score)
        return scores
