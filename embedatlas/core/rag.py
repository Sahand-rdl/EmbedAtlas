"""
EmbedAtlas — RAG / Search Engine

Three search modes
------------------
1. Keyword   — BM25 over stored documents
2. Semantic  — cosine similarity via ChromaDB + SentenceTransformer
3. Hybrid    — keyword hits first, semantic fills the rest

We do NOT pass embedding_function to ChromaDB's collection object —
same reason as embedder.py (conflict with persisted "default" fn).
Instead we encode the query ourselves and use collection.query() with
query_embeddings= instead of query_texts=.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from embedatlas.config import (
    CHROMA_DB_PATH,
    DEFAULT_TOP_K,
    EMBEDDING_MODELS,
    DEFAULT_MODEL_INDEX,
)


@dataclass
class SearchResult:
    id: str
    document: str
    metadata: Dict[str, Any]
    score: float
    match_type: str = "semantic"

    @property
    def snippet(self) -> str:
        return self.document[:300] + ("…" if len(self.document) > 300 else "")


class RAGEngine:
    def __init__(
        self,
        collection_name: str,
        model_id: str = EMBEDDING_MODELS[DEFAULT_MODEL_INDEX]["model_id"],
        db_path=CHROMA_DB_PATH,
        top_k: int = DEFAULT_TOP_K,
    ) -> None:
        import chromadb

        self.top_k = top_k
        self.collection_name = collection_name

        self._client = chromadb.PersistentClient(path=str(db_path))
        self._model = SentenceTransformer(model_id)

        # Get collection WITHOUT embedding_function to avoid conflict
        self._collection = self._client.get_collection(name=collection_name)

        # BM25 index — built lazily
        self._bm25 = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[dict] = []

    def _encode(self, text: str) -> List[float]:
        return self._model.encode(
            text, normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def search_semantic(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        k = top_k or self.top_k
        count = self._collection.count()
        if count == 0:
            return []

        query_emb = self._encode(query)
        kwargs: dict = dict(
            query_embeddings=[query_emb],
            n_results=min(k, count),
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        raw = self._collection.query(**kwargs)
        return self._parse_chroma(raw, "semantic")

    def search_keyword(
        self,
        query: str,
        top_k: Optional[int] = None,
    ) -> List[SearchResult]:
        k = top_k or self.top_k
        self._ensure_bm25()
        if not self._bm25_docs:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        max_s = scores.max()
        if max_s > 0:
            scores = scores / max_s

        top_idx = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_idx:
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
        k = top_k or self.top_k
        fetch_k = k * 3

        kw_results = self.search_keyword(query, top_k=fetch_k)
        sem_results = self.search_semantic(query, top_k=fetch_k, where=where)

        kw_map = {r.id: r for r in kw_results}
        sem_map = {r.id: r for r in sem_results}
        merged: Dict[str, SearchResult] = {}

        for rid, r in kw_map.items():
            if rid in sem_map:
                merged[rid] = SearchResult(
                    id=rid,
                    document=r.document,
                    metadata=r.metadata,
                    score=max(r.score, sem_map[rid].score),
                    match_type="both",
                )
            else:
                merged[rid] = r

        for rid, r in sem_map.items():
            if rid not in merged:
                merged[rid] = r

        priority = {"both": 0, "keyword": 1, "semantic": 2}
        return sorted(
            merged.values(),
            key=lambda r: (priority[r.match_type], -r.score),
        )[:k]

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: Optional[int] = None,
        where: Optional[Dict] = None,
    ) -> List[SearchResult]:
        mode = mode.lower()
        if mode == "keyword":
            return self.search_keyword(query, top_k=top_k)
        if mode == "semantic":
            return self.search_semantic(query, top_k=top_k, where=where)
        if mode == "hybrid":
            return self.search_hybrid(query, top_k=top_k, where=where)
        raise ValueError(f"Unknown mode: '{mode}'")

    def _ensure_bm25(self) -> None:
        if self._bm25 is not None:
            return
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError("pip install rank-bm25")

        total = self._collection.count()
        if total == 0:
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

        self._bm25 = BM25Okapi([_tokenize(d) for d in self._bm25_docs])

    def invalidate_bm25(self) -> None:
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_docs = []
        self._bm25_metas = []

    @staticmethod
    def _parse_chroma(raw: dict, match_type: str) -> List[SearchResult]:
        results = []
        if not raw or not raw.get("ids") or not raw["ids"][0]:
            return results
        for rid, doc, meta, dist in zip(
            raw["ids"][0], raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
        ):
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


def _tokenize(text: str) -> List[str]:
    import re

    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return [t for t in text.split() if len(t) > 1]
