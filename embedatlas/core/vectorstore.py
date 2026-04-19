"""
EmbedAtlas — VectorStore
All ChromaDB collection management: list, create, delete, inspect.
The Embedder handles writing; this module handles everything else.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from embedatlas.config import CHROMA_DB_PATH, CHROMA_DISTANCE_METRIC


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CollectionInfo:
    name: str
    count: int  # number of chunks stored
    model_id: Optional[str]  # embedding model used (stored in metadata)
    metadata: Dict[str, Any]  # raw ChromaDB collection metadata
    metadata_keys: List[str]  # unique metadata fields across all docs


# ---------------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------------


class VectorStore:
    """
    Thin wrapper around a ChromaDB PersistentClient.
    Provides collection-level CRUD and inspection helpers used by the UI.

    Parameters
    ----------
    db_path : path to the ChromaDB persistence directory
    """

    def __init__(self, db_path: Path | str = CHROMA_DB_PATH) -> None:
        self._db_path = Path(db_path)
        self._db_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._db_path))

    # ------------------------------------------------------------------
    # Collection listing & inspection
    # ------------------------------------------------------------------

    def list_collections(self) -> List[CollectionInfo]:
        """Return info for every collection in the database."""
        infos = []
        for col in self._client.list_collections():
            infos.append(self._inspect(col))
        return sorted(infos, key=lambda c: c.name)

    def collection_names(self) -> List[str]:
        return [c.name for c in self._client.list_collections()]

    def collection_exists(self, name: str) -> bool:
        return name in self.collection_names()

    def get_collection_info(self, name: str) -> CollectionInfo:
        col = self._client.get_collection(name=name)
        return self._inspect(col)

    def get_collection(self, name: str):
        """Return the raw ChromaDB collection object."""
        return self._client.get_collection(name=name)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        model_id: Optional[str] = None,
        extra_metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Create an empty collection.
        Raises ValueError if a collection with that name already exists.
        """
        if self.collection_exists(name):
            raise ValueError(f"Collection '{name}' already exists.")

        meta = {"hnsw:space": CHROMA_DISTANCE_METRIC}
        if model_id:
            meta["model_id"] = model_id
        if extra_metadata:
            meta.update(extra_metadata)

        return self._client.create_collection(name=name, metadata=meta)

    def delete_collection(self, name: str) -> None:
        """Permanently delete a collection and all its embeddings."""
        if not self.collection_exists(name):
            raise ValueError(f"Collection '{name}' does not exist.")
        self._client.delete_collection(name=name)

    def rename_collection(self, old_name: str, new_name: str) -> None:
        """
        ChromaDB doesn't support rename natively.
        We copy all data to a new collection then delete the old one.
        For large collections this can take a while — caller should warn the user.
        """
        if not self.collection_exists(old_name):
            raise ValueError(f"Collection '{old_name}' does not exist.")
        if self.collection_exists(new_name):
            raise ValueError(f"Collection '{new_name}' already exists.")

        old_col = self._client.get_collection(name=old_name)
        old_meta = dict(old_col.metadata or {})
        new_col = self._client.create_collection(name=new_name, metadata=old_meta)

        # Copy in batches
        batch_size = 500
        offset = 0
        while True:
            result = old_col.get(
                limit=batch_size,
                offset=offset,
                include=["embeddings", "documents", "metadatas"],
            )
            if not result["ids"]:
                break
            new_col.upsert(
                ids=result["ids"],
                embeddings=result["embeddings"],
                documents=result["documents"],
                metadatas=result["metadatas"],
            )
            offset += batch_size

        self._client.delete_collection(name=old_name)

    # ------------------------------------------------------------------
    # Data retrieval helpers (used by reduction.py and rag.py)
    # ------------------------------------------------------------------

    def get_all_embeddings(
        self,
        name: str,
        where: Optional[Dict] = None,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> Dict[str, Any]:
        """
        Retrieve embeddings + metadata + documents from a collection.

        Returns a dict with keys: ids, embeddings, documents, metadatas
        """
        col = self._client.get_collection(name=name)
        kwargs = dict(
            include=["embeddings", "documents", "metadatas"],
            offset=offset,
        )
        if where:
            kwargs["where"] = where
        if limit:
            kwargs["limit"] = limit

        return col.get(**kwargs)

    def get_embeddings_paginated(
        self,
        name: str,
        page_size: int = 5000,
        where: Optional[Dict] = None,
    ):
        """
        Generator that yields result dicts page by page.
        Use this for large collections to avoid loading everything into RAM.
        """
        offset = 0
        while True:
            result = self.get_all_embeddings(
                name,
                where=where,
                limit=page_size,
                offset=offset,
            )
            if not result["ids"]:
                break
            yield result
            if len(result["ids"]) < page_size:
                break
            offset += page_size

    def sample_embeddings(
        self,
        name: str,
        n: int,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Return a random sample of *n* items from the collection.
        Fetches sequentially then shuffles — suitable for viz use.
        """
        import random
        import numpy as np

        all_ids: List[str] = []
        all_embs: List = []
        all_docs: List[str] = []
        all_metas: List[dict] = []

        for page in self.get_embeddings_paginated(name, where=where):
            all_ids.extend(page["ids"])
            all_embs.extend(page["embeddings"])
            all_docs.extend(page["documents"])
            all_metas.extend(page["metadatas"])

        total = len(all_ids)
        if total == 0:
            return {"ids": [], "embeddings": [], "documents": [], "metadatas": []}

        n = min(n, total)
        indices = random.sample(range(total), n)

        return {
            "ids": [all_ids[i] for i in indices],
            "embeddings": [all_embs[i] for i in indices],
            "documents": [all_docs[i] for i in indices],
            "metadatas": [all_metas[i] for i in indices],
        }

    def get_unique_metadata_values(self, name: str, key: str) -> List[Any]:
        """
        Return all unique values for a given metadata key.
        Used to populate the 'color by' dropdown in the viz page.
        """
        all_values = []
        for page in self.get_embeddings_paginated(name, page_size=2000):
            for meta in page["metadatas"]:
                if key in meta:
                    all_values.append(meta[key])
        return list(dict.fromkeys(all_values))  # deduplicate, preserve order

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _inspect(self, col) -> CollectionInfo:
        """Build a CollectionInfo from a ChromaDB collection object."""
        count = col.count()
        meta = dict(col.metadata or {})

        # Discover metadata keys from a small sample
        meta_keys: List[str] = []
        if count > 0:
            sample = col.get(limit=min(count, 100), include=["metadatas"])
            keys = set()
            for m in sample.get("metadatas", []):
                keys.update(m.keys())
            # Exclude internal keys
            meta_keys = sorted(k for k in keys if not k.startswith("chunk_"))

        return CollectionInfo(
            name=col.name,
            count=count,
            model_id=meta.get("model_id"),
            metadata=meta,
            metadata_keys=meta_keys,
        )
