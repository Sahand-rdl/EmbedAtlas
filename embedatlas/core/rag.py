"""
EmbedAtlas — RAG / Search Engine

Three search modes
------------------
1. Keyword   — BM25 over stored documents; exact/fuzzy term matching
2. Semantic  — ChromaDB cosine similarity (vector search)
3. Hybrid    — Keyword hits first (ranked by BM25 score),
               then semantic-only hits not already in keyword results,
               ordered by relevance %. Each result carries a match-type badge.

Result schema (SearchResult dataclass)
---------------------------------------
id          : ChromaDB chunk id
document    : original chunk text
metadata    : dict of metadata fields
score       : float in [0, 1] — normalised relevance
match_type  : "keyword" | "semantic" | "both"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from embedatlas.config import (
    CHROMA_DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    DEFAULT_MODEL_INDEX,
)
from embedatlas.core.embedder import _make_embedding_fn


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    id: str
    document: str
    metadata: Dict[str, Any]
    score: float  # 0–1, higher = more relevant
    match_type: str = "semantic"  # "keyword" | "semantic" | "both"

    # Convenience: truncated snippet for display
    @property
    def snippet(self) -> str:
        return self.document[:300] + ("…" if len(self.document) > 300 else "")


# ---------------------------------------------------------------------------
# RAG engine
# ---------------------------------------------------------------------------


class RAGEngine:
    """
    Search engine backed by a ChromaDB collection.

    Parameters
    ----------
    collection_name : name of the ChromaDB collection to search
    model_id        : embedding model (must match the one used at ingestion)
    db_path         : ChromaDB persistence path
    top_k           : default number of results to return
    """

    def __init__(
        self,
        collection_name: str,
        model_id: str = EMBEDDING_MODELS[DEFAULT_MODEL_INDEX]["model_id"],
        db_path=CHROMA_DB_PATH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        import chromadb
        from chromadb.utils import embedding_functions

        self.top_k = top_k
        self.collection_name = collection_name
        self._model_id = model_id

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._embedding_fn = _make_embedding_fn(model_id)
        self._collection = self._client.get_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        # BM25 index — built lazily on first keyword/hybrid search
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[dict] = []

    # ------------------------------------------------------------------
    # Public search methods
    # ------------------------------------------------------------------

    def search_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """Pure vector similarity search."""
        k = top_k or self.top_k
        kwargs = dict(
            query_texts=[query],
            n_results=min(k, self._collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        return self._parse_chroma_results(raw, match_type="semantic")

    def search_keyword(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        """BM25 keyword search over all stored documents."""
        k = top_k or self.top_k
        self._ensure_bm25()

        if not self._bm25_docs:
            return []

        tokenized_query = _tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)

        # Normalise to [0, 1]
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score

        # Sort descending, take top-k
        top_indices = np.argsort(scores)[::-1][:k]

        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                break
            results.append(
                SearchResult(
                    id=self._bm25_ids[idx],
                    document=self._bm25_docs[idx],
                    metadata=self._bm25_metas[idx],
                    score=float(scores[idx]),
                    match_type="keyword",
                )
            )
        return results

    def search_hybrid(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Hybrid search: keyword hits first, then semantic-only fill.

        Logic:
        1. Run keyword search (BM25) → scored results
        2. Run semantic search → scored results
        3. Any chunk that appears in BOTH gets match_type='both'
           and its score is max(keyword_score, semantic_score)
        4. Keyword / 'both' results come first, sorted by score desc
        5. Pure-semantic results appended after, sorted by score desc
        6. Total list truncated at top_k
        """
        k = top_k or self.top_k
        fetch_k = k * 3  # over-fetch so we have enough after dedup

        kw_results = self.search_keyword(query, top_k=fetch_k)
        sem_results = self.search_semantic(query, top_k=fetch_k, where=where)

        kw_map = {r.id: r for r in kw_results}
        sem_map = {r.id: r for r in sem_results}

        merged: Dict[str, SearchResult] = {}

        # Keyword hits
        for rid, r in kw_map.items():
            if rid in sem_map:
                # Appears in both — elevate score, mark as 'both'
                merged[rid] = SearchResult(
                    id=rid,
                    document=r.document,
                    metadata=r.metadata,
                    score=max(r.score, sem_map[rid].score),
                    match_type="both",
                )
            else:
                merged[rid] = r

        # Semantic-only hits
        for rid, r in sem_map.items():
            if rid not in merged:
                merged[rid] = r

        # Sort: keyword/both first (by score), then semantic-only (by score)
        priority = {"both": 0, "keyword": 1, "semantic": 2}
        sorted_results = sorted(
            merged.values(),
            key=lambda r: (priority[r.match_type], -r.score),
        )

        return sorted_results[:k]

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        """
        Parameters
        ----------
        mode : "keyword" | "semantic" | "hybrid"
        """
        mode = mode.lower()
        if mode == "keyword":
            return self.search_keyword(query, top_k=top_k)
        if mode == "semantic":
            return self.search_semantic(query, top_k=top_k, where=where)
        if mode == "hybrid":
            return self.search_hybrid(query, top_k=top_k, where=where)
        raise ValueError(
            f"Unknown search mode: '{mode}'. Choose keyword/semantic/hybrid."
        )

    # ------------------------------------------------------------------
    # BM25 index management
    # ------------------------------------------------------------------

    def _ensure_bm25(self) -> None:
        """Build the BM25 index on first call (lazy)."""
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "rank-bm25 is required for keyword search. "
                "Install: pip install rank-bm25"
            )

        # Load all documents from ChromaDB
        total = self._collection.count()
        if total == 0:
            self._bm25 = None
            return

        batch_size = 5000
        offset = 0
        while True:
            result = self._collection.get(
                limit=batch_size,
                offset=offset,
                include=["documents", "metadatas"],
            )
            if not result["ids"]:
                break
            self._bm25_ids.extend(result["ids"])
            self._bm25_docs.extend(result["documents"])
            self._bm25_metas.extend(result["metadatas"])
            if len(result["ids"]) < batch_size:
                break
            offset += batch_size

        tokenized_corpus = [_tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized_corpus)

    def invalidate_bm25(self) -> None:
        """Call this after adding new documents so the index is rebuilt."""
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_docs = []
        self._bm25_metas = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_chroma_results(raw: dict, match_type: str) -> List[SearchResult]:
        """Convert raw ChromaDB query output to SearchResult list."""
        results = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results

        ids = raw["ids"][0]
        docs = raw["documents"][0]
        metas = raw["metadatas"][0]
        distances = raw["distances"][0]

        for rid, doc, meta, dist in zip(ids, docs, metas, distances):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite
            # Convert to similarity score in [0, 1]
            score = max(0.0, 1.0 - dist / 2.0)
            results.append(
                SearchResult(
                    id=rid,
                    document=doc,
                    metadata=meta,
                    score=score,
                    match_type=match_type,
                )
            )
        return results


# ---------------------------------------------------------------------------
# Tokeniser (shared by BM25 and hybrid)
# ---------------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    """Simple whitespace + lowercase tokeniser for BM25."""
    import re

    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]
