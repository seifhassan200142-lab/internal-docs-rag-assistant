from __future__ import annotations

from app.schemas.models import RetrievedChunk

INSUFFICIENT_CONTEXT_MESSAGE = "I could not find enough information in the uploaded documents to answer this reliably."


def build_context(chunks: list[RetrievedChunk]) -> str:
    context_blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        page = f"page {chunk.page_number}" if chunk.page_number is not None else "page unavailable"
        context_blocks.append(
            f"[Source {index}] document={chunk.document_name}; {page}; chunk_id={chunk.chunk_id}\n{chunk.text}"
        )
    return "\n\n".join(context_blocks)


def build_rag_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict[str, str]]:
    context = build_context(chunks)
    system_prompt = (
        "You are a careful internal document assistant. Answer only using the provided context. "
        "If the context is not enough, reply exactly: "
        f"'{INSUFFICIENT_CONTEXT_MESSAGE}' "
        "Do not use outside knowledge. Be concise and cite relevant source numbers inline like [Source 1]."
    )
    user_prompt = f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
