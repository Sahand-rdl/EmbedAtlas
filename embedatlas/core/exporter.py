"""
EmbedAtlas — Exporter
Converts search results and embedding samples to downloadable formats.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List

from embedatlas.core.rag import SearchResult


# ---------------------------------------------------------------------------
# Search results → CSV / JSON
# ---------------------------------------------------------------------------


def results_to_csv(results: List[SearchResult]) -> str:
    """Return a UTF-8 CSV string of search results."""
    if not results:
        return ""

    output = io.StringIO()
    fieldnames = ["rank", "score", "match_type", "id", "snippet", "document"] + sorted(
        {k for r in results for k in r.metadata}
    )

    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for rank, r in enumerate(results, start=1):
        row = {
            "rank": rank,
            "score": f"{r.score:.4f}",
            "match_type": r.match_type,
            "id": r.id,
            "snippet": r.snippet,
            "document": r.document,
            **r.metadata,
        }
        writer.writerow(row)

    return output.getvalue()


def results_to_json(results: List[SearchResult]) -> str:
    """Return a pretty-printed JSON string of search results."""
    payload = [
        {
            "rank": rank,
            "score": round(r.score, 4),
            "match_type": r.match_type,
            "id": r.id,
            "document": r.document,
            "metadata": r.metadata,
        }
        for rank, r in enumerate(results, start=1)
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Embedding sample → CSV (for external analysis)
# ---------------------------------------------------------------------------


def sample_to_csv(sample: Dict[str, Any]) -> str:
    """
    Export a VectorStore sample (ids, documents, metadatas) as CSV.
    Embeddings are excluded by default (they are huge and rarely useful as CSV).
    """
    ids = sample.get("ids", [])
    documents = sample.get("documents", [])
    metadatas = sample.get("metadatas", [])

    if not ids:
        return ""

    meta_keys = sorted({k for m in metadatas for k in m})
    fieldnames = ["id", "document"] + meta_keys

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    for rid, doc, meta in zip(ids, documents, metadatas):
        row = {"id": rid, "document": doc[:500], **meta}
        writer.writerow(row)

    return output.getvalue()


def sample_to_json(sample: Dict[str, Any], include_embeddings: bool = False) -> str:
    """Export a VectorStore sample as JSON."""
    ids = sample.get("ids", [])
    documents = sample.get("documents", [])
    metadatas = sample.get("metadatas", [])
    embeddings = sample.get("embeddings", []) if include_embeddings else []

    records = []
    for i, (rid, doc, meta) in enumerate(zip(ids, documents, metadatas)):
        rec: Dict[str, Any] = {"id": rid, "document": doc, "metadata": meta}
        if include_embeddings and i < len(embeddings):
            rec["embedding"] = embeddings[i]
        records.append(rec)

    return json.dumps(records, ensure_ascii=False, indent=2)
