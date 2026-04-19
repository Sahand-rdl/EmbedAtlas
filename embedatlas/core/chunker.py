"""
EmbedAtlas — Chunker
Wraps LangChain's RecursiveCharacterTextSplitter with sensible defaults
and a clean interface used by ingestion.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from embedatlas.config import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    MIN_CHUNK_SIZE,
    MAX_CHUNK_SIZE,
)


@dataclass
class Chunk:
    """A single chunk of text with its source metadata."""

    text: str
    index: int  # chunk index within its source document
    doc_id: str  # identifier of the source document
    metadata: dict = field(default_factory=dict)  # arbitrary user metadata


class Chunker:
    """
    Splits raw text into overlapping chunks using LangChain's
    RecursiveCharacterTextSplitter (sentence-aware, then word-aware fallback).

    Parameters
    ----------
    chunk_size : int
        Target character length per chunk.
    chunk_overlap : int
        Number of characters shared between consecutive chunks.
    """

    def __init__(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ) -> None:
        chunk_size = max(MIN_CHUNK_SIZE, min(chunk_size, MAX_CHUNK_SIZE))
        chunk_overlap = max(0, min(chunk_overlap, chunk_size // 2))

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # Try to split on paragraph → sentence → word → character
            separators=["\n\n", "\n", ". ", "! ", "? ", " ", ""],
            length_function=len,
            is_separator_regex=False,
        )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split(
        self,
        text: str,
        doc_id: str,
        metadata: dict | None = None,
    ) -> List[Chunk]:
        """
        Split *text* into Chunk objects.

        Parameters
        ----------
        text     : raw string content of the document
        doc_id   : a stable identifier for the source document
                   (e.g. filename, URL, HF dataset row id)
        metadata : arbitrary key-value pairs carried on every chunk
                   (e.g. {"source": "arxiv", "label": "en"})

        Returns
        -------
        List[Chunk] — at least one chunk even for very short texts.
        """
        if not text or not text.strip():
            return []

        metadata = metadata or {}
        raw_chunks: List[str] = self._splitter.split_text(text)

        return [
            Chunk(
                text=chunk,
                index=i,
                doc_id=doc_id,
                metadata={**metadata, "chunk_index": i, "doc_id": doc_id},
            )
            for i, chunk in enumerate(raw_chunks)
            if chunk.strip()  # skip whitespace-only artefacts
        ]

    def split_many(
        self,
        documents: List[dict],
    ) -> List[Chunk]:
        """
        Convenience method for a list of document dicts.

        Each dict must have:
            "text"    : str  — raw content
            "doc_id"  : str  — unique source identifier
            "metadata": dict — optional, merged onto every chunk

        Returns a flat list of all chunks across all documents.
        """
        all_chunks: List[Chunk] = []
        for doc in documents:
            chunks = self.split(
                text=doc["text"],
                doc_id=doc["doc_id"],
                metadata=doc.get("metadata", {}),
            )
            all_chunks.extend(chunks)
        return all_chunks
