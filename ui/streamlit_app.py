from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

API_URL = os.getenv("STREAMLIT_API_URL", "http://localhost:8000")

st.set_page_config(page_title="Internal Docs RAG Assistant", page_icon="📚", layout="wide")
st.title("📚 Internal Docs RAG Assistant")
st.caption("Upload internal documents, ask questions, and inspect grounded citations.")


def api_get(path: str) -> Any:
    response = requests.get(f"{API_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def api_post(path: str, **kwargs: Any) -> Any:
    response = requests.post(f"{API_URL}{path}", timeout=120, **kwargs)
    response.raise_for_status()
    return response.json()


def api_delete(path: str) -> Any:
    response = requests.delete(f"{API_URL}{path}", timeout=60)
    response.raise_for_status()
    return response.json()


def _format_score(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.4f}"


with st.sidebar:
    st.header("Connection")
    st.write(f"API: `{API_URL}`")
    try:
        health = api_get("/health")
        st.success(f"API status: {health['status']}")
    except Exception as exc:
        st.error(f"API is not reachable: {exc}")

    st.header("Retrieval Settings")
    top_k = st.slider("Top K chunks", min_value=1, max_value=20, value=6)
    include_chunks = st.checkbox("Show retrieved chunks", value=True)


tab_upload, tab_query, tab_documents = st.tabs(["Upload", "Ask", "Documents"])

with tab_upload:
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Supported formats: PDF, TXT, Markdown, CSV",
        type=["pdf", "txt", "md", "markdown", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents", disabled=not uploaded_files):
        for uploaded_file in uploaded_files or []:
            with st.spinner(f"Indexing {uploaded_file.name}..."):
                try:
                    result = api_post(
                        "/api/ingest",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    )
                    st.success(f"{result['document_name']} indexed with {result['chunks_indexed']} chunks.")
                except requests.HTTPError as exc:
                    detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                    st.error(f"Failed to index {uploaded_file.name}: {detail}")
                except Exception as exc:
                    st.error(f"Failed to index {uploaded_file.name}: {exc}")

with tab_query:
    st.subheader("Ask a question")
    question = st.text_area("Question", placeholder="Example: What is the remote work policy?")

    if st.button("Ask", disabled=not question.strip()):
        with st.spinner("Retrieving context and generating answer..."):
            try:
                result = api_post(
                    "/api/query",
                    json={"question": question, "top_k": top_k, "include_chunks": include_chunks},
                )
                st.markdown("### Answer")
                st.write(result["answer"])

                st.markdown("### Citations")
                if result["citations"]:
                    for citation in result["citations"]:
                        page = citation["page_number"] if citation["page_number"] is not None else "N/A"
                        with st.expander(f"{citation['document_name']} | page {page} | {citation['chunk_id']}"):
                            st.write(citation["preview"])
                else:
                    st.info("No citations were returned.")

                if include_chunks:
                    st.markdown("### Retrieved chunks")
                    for chunk in result["retrieved_chunks"]:
                        title = f"{chunk['document_name']} | {chunk['chunk_id']}"
                        with st.expander(title):
                            score_cols = st.columns(4)
                            score_cols[0].metric("Dense", _format_score(chunk.get("dense_score")))
                            score_cols[1].metric("BM25", _format_score(chunk.get("sparse_score")))
                            score_cols[2].metric("RRF", _format_score(chunk.get("fusion_score")))
                            score_cols[3].metric("Rerank", _format_score(chunk.get("rerank_score")))
                            st.write(chunk["text"])
            except requests.HTTPError as exc:
                detail = exc.response.json().get("detail", str(exc)) if exc.response is not None else str(exc)
                st.error(detail)
            except Exception as exc:
                st.error(f"Query failed: {exc}")

with tab_documents:
    st.subheader("Indexed documents")
    try:
        documents = api_get("/api/documents")
        if not documents:
            st.info("No documents are indexed yet.")
        for document in documents:
            with st.container(border=True):
                st.write(f"**{document['document_name']}**")
                st.write(f"Document ID: `{document['document_id']}`")
                st.write(f"Chunks: {document['chunk_count']}")
                st.write(f"Source path: `{document['source_file_path']}`")
                if st.button("Delete", key=f"delete-{document['document_id']}"):
                    try:
                        api_delete(f"/api/documents/{document['document_id']}")
                        st.success("Document deleted. Refresh the page to update the list.")
                    except Exception as exc:
                        st.error(f"Delete failed: {exc}")
    except Exception as exc:
        st.error(f"Could not load documents: {exc}")
