"""
EmbedAtlas — Embedder
Wraps SentenceTransformers to embed chunks and store them in ChromaDB.

Key design decision
-------------------
We do NOT pass an embedding_function to ChromaDB's collection object.
ChromaDB stores which embedding function was used in collection metadata
and will reject any new one that differs — causing the "conflict" error.
Instead we call the ST model directly and pass raw float vectors to upsert().
This gives us full control and zero conflicts.
"""

from __future__ import annotations

import uuid
from typing import Callable, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from embedatlas.config import (
    CHROMA_DB_PATH,
    CHROMA_DISTANCE_METRIC,
    EMBEDDING_MODELS,
    DEFAULT_MODEL_INDEX,
)
from embedatlas.core.chunker import Chunk


def _make_embedding_fn(model_id: str) -> SentenceTransformer:
    """Load a SentenceTransformer model. Cached by Python's module system."""
    return SentenceTransformer(model_id)


class Embedder:
    """
    Embeds chunks and upserts them into a ChromaDB collection.

    Parameters
    ----------
    collection_name : target ChromaDB collection
    model_id        : SentenceTransformer model identifier
    db_path         : ChromaDB persistence directory
    batch_size      : chunks per forward pass (reduce on OOM)
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

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._model = _make_embedding_fn(model_id)

        # Get or create collection — NO embedding_function argument
        # to avoid ChromaDB's embedding function conflict error.
        existing = [c.name for c in self._client.list_collections()]
        if collection_name in existing:
            self._collection = self._client.get_collection(name=collection_name)
        else:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": CHROMA_DISTANCE_METRIC,
                    "model_id": model_id,
                },
            )

    @property
    def collection(self):
        return self._collection

    @property
    def count(self) -> int:
        return self._collection.count()

    def embed_chunks(
        self,
        chunks: List[Chunk],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> int:
        """Encode chunks and upsert into ChromaDB. Returns count stored."""
        if not chunks:
            return 0

        total = len(chunks)
        stored = 0

        for batch_start in range(0, total, self.batch_size):
            batch = chunks[batch_start : batch_start + self.batch_size]
            texts = [c.text for c in batch]
            ids = [self._make_id(c) for c in batch]
            metadatas = [c.metadata for c in batch]

            # Encode directly — returns numpy array, convert to list of lists
            embeddings = self._model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

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
        """Embed raw strings directly (no Chunk objects)."""
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
                f"{doc_ids[j]}_{i + k}"
                for k, j in enumerate(range(i, min(i + self.batch_size, total)))
            ]

            embeddings = self._model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

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

    @staticmethod
    def _make_id(chunk: Chunk) -> str:
        safe_doc = chunk.doc_id.replace("/", "_").replace(" ", "_")
        return f"{safe_doc}_chunk_{chunk.index}"


def get_model_options() -> List[dict]:
    return EMBEDDING_MODELS


def model_id_from_display(display_name: str) -> str:
    for m in EMBEDDING_MODELS:
        if m["display_name"] == display_name:
            return m["model_id"]
    raise ValueError(f"Unknown model display name: {display_name}")
