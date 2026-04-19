"""
EmbedAtlas — Ingestion
Handles all data sources and converts them to a flat list of
{"text": str, "doc_id": str, "metadata": dict} dicts ready for chunking.

Supported sources
-----------------
1. Local files  : .txt .md .rst .pdf .csv .tsv .json .jsonl
2. Local folder : recursively finds all supported files
3. HF dataset   : load_dataset(repo_id, split=..., streaming=True)
4. GitHub / raw URL : streams a single text/csv/json file over HTTP
"""

from __future__ import annotations


import csv
import io
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Generator, List, Optional
from urllib.parse import urlparse

import requests

from embedatlas.config import (
    ALL_SUPPORTED_EXTENSIONS,
    DEFAULT_PDF_PARSER,
    HF_DEFAULT_MAX_ROWS,
    HF_DEFAULT_TEXT_COLUMN,
    HF_MAX_ROWS_HARD_CAP,
    SUPPORTED_EXTENSIONS,
    TEMP_DIR,
)

# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------
Document = dict  # {"text": str, "doc_id": str, "metadata": dict}


# ---------------------------------------------------------------------------
# PDF parsers
# ---------------------------------------------------------------------------


def _parse_pdf_pymupdf(path: Path) -> str:
    """Extract plain text from a PDF using PyMuPDF (fast, lightweight)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF parsing. Install it with:\n"
            "  pip install pymupdf"
        )
    doc = fitz.open(str(path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n\n".join(pages)


def _parse_pdf_docling(path: Path) -> str:
    """Extract structured text from a PDF using Docling (heavy, layout-aware)."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError:
        raise ImportError(
            "Docling is required for high-quality PDF parsing. Install it with:\n"
            "  pip install docling"
        )
    converter = DocumentConverter()
    result = converter.convert(str(path))
    doc = result.document
    try:
        data = doc.export_to_dict()
        texts = data.get("texts", []) or data.get("content", [])
        return " ".join(
            item.get("text", "") for item in texts if isinstance(item, dict)
        )
    except Exception:
        # Fallback: raw markdown export
        return doc.export_to_markdown()


def _parse_pdf(path: Path, parser: str = DEFAULT_PDF_PARSER) -> str:
    if parser == "docling":
        return _parse_pdf_docling(path)
    return _parse_pdf_pymupdf(path)


# ---------------------------------------------------------------------------
# Single-file loaders
# ---------------------------------------------------------------------------


def _load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_csv_file(path: Path, text_column: Optional[str] = None) -> str:
    """
    If *text_column* is provided, join that column's values.
    Otherwise concatenate all columns row by row.
    """
    rows = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if text_column and text_column in row:
                rows.append(row[text_column])
            else:
                rows.append(" | ".join(str(v) for v in row.values()))
    return "\n".join(rows)


def _load_json_file(path: Path, text_column: Optional[str] = None) -> str:
    """
    Handles both JSON arrays and JSONL (one object per line).
    Extracts *text_column* if specified, otherwise serialises each record.
    """
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    records: list = []

    if path.suffix == ".jsonl" or (text and text[0] == "{" and "\n" in text):
        # JSONL
        for line in text.splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    else:
        data = json.loads(text)
        records = data if isinstance(data, list) else [data]

    parts = []
    for rec in records:
        if isinstance(rec, dict):
            if text_column and text_column in rec:
                parts.append(str(rec[text_column]))
            else:
                parts.append(json.dumps(rec, ensure_ascii=False))
        else:
            parts.append(str(rec))
    return "\n".join(parts)


def _load_local_file(
    path: Path,
    pdf_parser: str = DEFAULT_PDF_PARSER,
    text_column: Optional[str] = None,
) -> str:
    suffix = path.suffix.lower()
    if suffix in SUPPORTED_EXTENSIONS["text"]:
        return _load_text_file(path)
    if suffix in SUPPORTED_EXTENSIONS["pdf"]:
        return _parse_pdf(path, parser=pdf_parser)
    if suffix in SUPPORTED_EXTENSIONS["table"]:
        return _load_csv_file(path, text_column=text_column)
    if suffix in SUPPORTED_EXTENSIONS["json"]:
        return _load_json_file(path, text_column=text_column)
    raise ValueError(f"Unsupported file extension: {suffix}")


# ---------------------------------------------------------------------------
# Public ingestion functions
# ---------------------------------------------------------------------------


def ingest_local_file(
    path: str | Path,
    label: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    pdf_parser: str = DEFAULT_PDF_PARSER,
    text_column: Optional[str] = None,
) -> List[Document]:
    """
    Load a single local file into a list of Documents.

    Parameters
    ----------
    path           : path to the file
    label          : user-defined category label (stored in metadata)
    extra_metadata : any additional key-value pairs to attach
    pdf_parser     : "pymupdf" or "docling"
    text_column    : for CSV/JSON, which column contains the main text
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    text = _load_local_file(path, pdf_parser=pdf_parser, text_column=text_column)
    if not text.strip():
        return []

    metadata = {
        "source": str(path),
        "filename": path.name,
        **({"label": label} if label else {}),
        **(extra_metadata or {}),
    }
    return [{"text": text, "doc_id": path.name, "metadata": metadata}]


def ingest_local_folder(
    folder: str | Path,
    label: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    pdf_parser: str = DEFAULT_PDF_PARSER,
    text_column: Optional[str] = None,
    recursive: bool = True,
) -> List[Document]:
    """
    Recursively load all supported files from a local folder.
    Each file becomes one Document (chunking happens downstream).
    """
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a directory: {folder}")

    pattern = "**/*" if recursive else "*"
    docs: List[Document] = []
    for path in sorted(folder.glob(pattern)):
        if path.is_file() and path.suffix.lower() in ALL_SUPPORTED_EXTENSIONS:
            try:
                docs.extend(
                    ingest_local_file(
                        path,
                        label=label,
                        extra_metadata=extra_metadata,
                        pdf_parser=pdf_parser,
                        text_column=text_column,
                    )
                )
            except Exception as e:
                # Non-fatal: log and continue
                print(f"[Ingestion] Skipping {path.name}: {e}")
    return docs


def ingest_hf_dataset(
    repo_id: str,
    split: str = "train",
    text_column: str = HF_DEFAULT_TEXT_COLUMN,
    label_column: Optional[str] = None,
    max_rows: int = HF_DEFAULT_MAX_ROWS,
    extra_metadata: Optional[dict] = None,
    progress_callback=None,  # callable(current_row: int) → None
) -> List[Document]:
    """
    Stream a HuggingFace dataset and return Documents.

    Parameters
    ----------
    repo_id        : e.g. "allenai/c4", "wikitext", "imdb"
    split          : "train" | "test" | "validation" | ...
    text_column    : column that contains the raw text
    label_column   : optional column to use as the 'label' metadata field
    max_rows       : hard cap (capped again at HF_MAX_ROWS_HARD_CAP)
    progress_callback : optional callable for UI progress updates
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "The 'datasets' library is required for HuggingFace ingestion.\n"
            "  pip install datasets"
        )

    max_rows = min(max_rows, HF_MAX_ROWS_HARD_CAP)

    dataset = load_dataset(repo_id, split=split, streaming=True, trust_remote_code=True)

    docs: List[Document] = []
    for i, row in enumerate(dataset):
        if i >= max_rows:
            break

        text = str(row.get(text_column, "")).strip()
        if not text:
            continue

        label = str(row[label_column]) if label_column and label_column in row else None
        row_id = f"{repo_id.replace('/', '_')}_{split}_{i}"
        metadata = {
            "source": repo_id,
            "split": split,
            "row_index": i,
            **({"label": label} if label else {}),
            **(extra_metadata or {}),
        }
        docs.append({"text": text, "doc_id": row_id, "metadata": metadata})

        if progress_callback:
            progress_callback(i + 1)

    return docs


def ingest_url(
    url: str,
    label: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
    text_column: Optional[str] = None,
    pdf_parser: str = DEFAULT_PDF_PARSER,
    timeout: int = 30,
) -> List[Document]:
    """
    Download a raw file from a URL (GitHub raw, HF raw, any HTTP/S link)
    and return Documents.

    Supports: .txt .md .rst .csv .tsv .json .jsonl .pdf
    """
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "downloaded_file"
    suffix = Path(parsed.path).suffix.lower()

    if suffix not in ALL_SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"URL points to an unsupported file type: '{suffix}'\n"
            f"Supported: {ALL_SUPPORTED_EXTENSIONS}"
        )

    # Stream download into a temp file
    TEMP_DIR.mkdir(parents=True, exist_ok=True)
    tmp_path = TEMP_DIR / f"{uuid.uuid4().hex}_{filename}"

    try:
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        text = _load_local_file(
            tmp_path, pdf_parser=pdf_parser, text_column=text_column
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    if not text.strip():
        return []

    metadata = {
        "source": url,
        "filename": filename,
        **({"label": label} if label else {}),
        **(extra_metadata or {}),
    }
    return [{"text": text, "doc_id": filename, "metadata": metadata}]
