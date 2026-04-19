"""
EmbedAtlas — Embedder
Wraps SentenceTransformers + ChromaDB's embedding function to embed
chunks and store them in a named collection.

Design notes
------------
- Uses ChromaDB's built-in SentenceTransformerEmbeddingFunction so that
  ChromaDB handles batching internally during queries.
- For *ingestion* we call model.encode() directly so we can report
  progress chunk-by-chunk to the UI.
- The original chunk text is always stored as ChromaDB `documents` so
  hover-tooltips and search snippets never need a second lookup.
"""

from __future__ import annotations

import uuid
from typing import Callable, List, Optional

import chromadb
from chromadb.utils import embedding_functions

from embedatlas.config import (
    CHROMA_DB_PATH,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_MODELS,
    DEFAULT_MODEL_INDEX,
)
from embedatlas.core.chunker import Chunk


# ---------------------------------------------------------------------------
# Helper: build the ChromaDB embedding function for a given model
# ---------------------------------------------------------------------------


def _make_embedding_fn(model_id: str):
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_id,
        # normalize_embeddings=True is important for cosine distance
        normalize_embeddings=True,
    )


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------


class Embedder:
    """
    Embeds a list of Chunks and upserts them into a ChromaDB collection.

    Parameters
    ----------
    collection_name : target ChromaDB collection
    model_id        : SentenceTransformer model identifier
    db_path         : path to the ChromaDB persistence directory
    batch_size      : number of chunks encoded per forward pass
                      (reduce if you hit OOM errors)
    """

    def __init__(
        self,
        collection_name: str,
        model_id: str = EMBEDDING_MODELS[DEFAULT_MODEL_INDEX]["model_id"],
        db_path=CHROMA_DB_PATH,
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.batch_size = batch_size
        self.collection_name = collection_name

        # ChromaDB persistent client
        self._client = chromadb.PersistentClient(path=str(db_path))

        # Embedding function (also used by ChromaDB for query-time encoding)
        self._embedding_fn = _make_embedding_fn(model_id)

        # Get or create the collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def collection(self):
        return self._collection

    @property
    def count(self) -> int:
        """Number of chunks currently stored in the collection."""
        return self._collection.count()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Encode *chunks* and upsert them into the collection.

        Parameters
        ----------
        chunks            : list of Chunk objects (from chunker.py)
        progress_callback : callable(done: int, total: int) → None
                            called after each batch for UI progress bars

        Returns
        -------
        Number of chunks successfully stored.
        """
        if not chunks:
            return 0

        total = len(chunks)
        stored = 0

        for batch_start in range(0, total, self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]

            texts = [c.text for c in batch]
            ids = [self._make_id(c) for c in batch]
            metadatas = [c.metadata for c in batch]

            # Encode with the underlying ST model directly so we can
            # report per-batch progress. ChromaDB's embedding_fn is still
            # used at query time.
            embeddings = self._embedding_fn(texts)  # list of lists

            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            stored += len(batch)
            if progress_callback:
                progress_callback(stored, total)

        return stored

    def embed_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        doc_ids: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """
        Lower-level method: embed raw strings directly (no Chunk objects).
        Useful for HF dataset rows that arrive one-at-a-time.
        """
        if not texts:
            return 0

        metadatas = metadatas or [{} for _ in texts]
        doc_ids = doc_ids or [str(uuid.uuid4()) for _ in texts]

        total = len(texts)
        stored = 0

        for i in range(0, total, self.batch_size):
            batch_texts = texts[i : i + self.batch_size]
            batch_meta = metadatas[i : i + self.batch_size]
            batch_ids = [
                f"{doc_ids[j]}_{i+k}"
                for k, j in enumerate(range(i, min(i + self.batch_size, total)))
            ]

            embeddings = self._embedding_fn(batch_texts)

            self._collection.upsert(
                ids=batch_ids,
                embeddings=embeddings,
                documents=batch_texts,
                metadatas=batch_meta,
            )

            stored += len(batch_texts)
            if progress_callback:
                progress_callback(stored, total)

        return stored

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        """
        Deterministic chunk ID: <doc_id>_chunk_<index>
        Upsert means re-ingesting the same file won't create duplicates.
        """
        safe_doc = chunk.doc_id.replace("/", "_").replace(" ", "_")
        return f"{safe_doc}_chunk_{chunk.index}"


# ---------------------------------------------------------------------------
# Convenience: list available models for the UI dropdown
# ---------------------------------------------------------------------------


def get_model_options() -> List[dict]:
    """Return the full EMBEDDING_MODELS registry for UI display."""
    return EMBEDDING_MODELS


def model_id_from_display(display_name: str) -> str:
    """Reverse-lookup model_id from the dropdown display name."""
    for m in EMBEDDING_MODELS:
        if m["display_name"] == display_name:
            return m["model_id"]
    raise ValueError(f"Unknown model display name: {display_name}")
