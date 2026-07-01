from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class LoadedPage:
    text: str
    page_number: Optional[int] = None


@dataclass(frozen=True)
class LoadedDocument:
    document_name: str
    source_file_path: str
    pages: list[LoadedPage]


SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv"}


class UnsupportedFileTypeError(ValueError):
    pass


def load_document(file_path: Path) -> LoadedDocument:
    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'. Supported types: {allowed}")

    if suffix == ".pdf":
        pages = _load_pdf(file_path)
    elif suffix in {".txt", ".md", ".markdown"}:
        pages = [LoadedPage(text=file_path.read_text(encoding="utf-8"), page_number=None)]
    elif suffix == ".csv":
        pages = [LoadedPage(text=_load_csv_as_text(file_path), page_number=None)]
    else:
        raise UnsupportedFileTypeError(f"Unsupported file type '{suffix}'.")

    return LoadedDocument(
        document_name=file_path.name,
        source_file_path=str(file_path),
        pages=[page for page in pages if page.text.strip()],
    )


def _load_pdf(file_path: Path) -> list[LoadedPage]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError("PyMuPDF is required for PDF ingestion. Install requirements.txt first.") from exc

    pages: list[LoadedPage] = []
    with fitz.open(file_path) as document:
        for page_index, page in enumerate(document, start=1):
            text = page.get_text("text")
            if text.strip():
                pages.append(LoadedPage(text=text, page_number=page_index))
    return pages


def _load_csv_as_text(file_path: Path) -> str:
    lines: list[str] = []
    with file_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames:
            lines.append("CSV columns: " + ", ".join(reader.fieldnames))
        for row_number, row in enumerate(reader, start=1):
            values = [f"{key}: {value}" for key, value in row.items()]
            lines.append(f"Row {row_number}: " + " | ".join(values))
    return "\n".join(lines)
