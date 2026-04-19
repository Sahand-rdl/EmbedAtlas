"""
EmbedAtlas — Dimensionality Reduction
PCA / UMAP / t-SNE → Plotly interactive scatter figures.

Design
------
- Takes a sample dict (ids, embeddings, documents, metadatas) from VectorStore
- Reduces to 2D
- Returns a Plotly Figure with hover text showing the original chunk text
- Centroid-per-label fallback for very large samples
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from embedatlas.config import (
    CENTROID_FALLBACK_FRACTION,
    CENTROID_FALLBACK_MIN_DOCS,
    PCA_DEFAULTS,
    REDUCTION_MAX_POINTS,
    TSNE_DEFAULTS,
    UMAP_DEFAULTS,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_array(embeddings: List[List[float]]) -> np.ndarray:
    return np.array(embeddings, dtype=np.float32)


def _run_pca(X: np.ndarray) -> np.ndarray:
    from sklearn.decomposition import PCA

    pca = PCA(n_components=2, **PCA_DEFAULTS)
    return pca.fit_transform(X)


def _run_umap(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    try:
        import umap
    except ImportError:
        raise ImportError("umap-learn is required. Install: pip install umap-learn")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=UMAP_DEFAULTS["random_state"],
    )
    return reducer.fit_transform(X)


def _run_tsne(
    X: np.ndarray,
    perplexity: int,
    max_iter: int,
) -> np.ndarray:
    from sklearn.manifold import TSNE

    # perplexity must be < n_samples
    perplexity = min(perplexity, max(2, X.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=TSNE_DEFAULTS["learning_rate"],
        init=TSNE_DEFAULTS["init"],
        random_state=TSNE_DEFAULTS["random_state"],
        max_iter=max_iter,
        method=TSNE_DEFAULTS["method"],
        angle=TSNE_DEFAULTS["angle"],
    )
    return tsne.fit_transform(X)


def _centroids_per_label(
    X: np.ndarray,
    labels: List[str],
    ids: List[str],
    documents: List[str],
    metadatas: List[dict],
) -> tuple[np.ndarray, List[str], List[str], List[str], List[dict]]:
    """
    Collapse embeddings to one centroid per unique label.
    Returns (X_centroids, ids_out, labels_out, docs_out, metas_out).
    """
    label_arr = np.array(labels)
    unique = list(dict.fromkeys(labels))  # preserve order

    X_c, ids_c, labs_c, docs_c, metas_c = [], [], [], [], []
    for lab in unique:
        mask = label_arr == lab
        centroid = X[mask].mean(axis=0)
        X_c.append(centroid)
        ids_c.append(f"centroid_{lab}")
        labs_c.append(lab)
        docs_c.append(f"[Centroid of {mask.sum()} chunks — label: {lab}]")
        # merge representative metadata
        first_meta = next(m for m, flag in zip(metadatas, mask) if flag)
        metas_c.append({**first_meta, "centroid": True, "n_points": int(mask.sum())})

    return np.stack(X_c), ids_c, labs_c, docs_c, metas_c


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def reduce_and_plot(
    sample: Dict[str, Any],
    method: str = "UMAP",
    color_by: Optional[str] = None,  # metadata key for colouring
    label_key: Optional[str] = None,  # alias for color_by (kept for clarity)
    # t-SNE params
    tsne_perplexity: int = TSNE_DEFAULTS["perplexity"],
    tsne_max_iter: int = TSNE_DEFAULTS["max_iter"],
    # UMAP params
    umap_n_neighbors: int = UMAP_DEFAULTS["n_neighbors"],
    umap_min_dist: float = UMAP_DEFAULTS["min_dist"],
    # Display
    point_size: int = 6,
    point_opacity: float = 0.75,
    dark_mode: bool = False,
) -> tuple[go.Figure, bool, str]:
    """
    Reduce embeddings and return a Plotly figure.

    Parameters
    ----------
    sample      : dict from VectorStore.sample_embeddings(...)
                  keys: ids, embeddings, documents, metadatas
    method      : "PCA" | "UMAP" | "t-SNE"
    color_by    : metadata field to use for point colours
    label_key   : same as color_by (one of the two may be provided)

    Returns
    -------
    (fig, used_centroids, info_message)
    """
    color_by = color_by or label_key

    ids = sample["ids"]
    embeddings = sample["embeddings"]
    documents = sample["documents"]
    metadatas = sample["metadatas"]

    if not ids:
        raise ValueError("Sample is empty — nothing to visualise.")

    X = _to_array(embeddings)

    # -------------------------------------------------------------------
    # Centroid fallback
    # -------------------------------------------------------------------
    used_centroids = False
    info_message = ""

    total_in_collection = len(ids)  # post-sample; ratio check done upstream
    # Centroid logic is triggered by the UI before calling this function,
    # but we also have a safety check here.
    labels_raw = [m.get(color_by, "unknown") if color_by else "all" for m in metadatas]

    max_pts = REDUCTION_MAX_POINTS.get(method)
    if max_pts and X.shape[0] > max_pts:
        # Hard cap: collapse to centroids
        X, ids, labels_raw, documents, metadatas = _centroids_per_label(
            X, labels_raw, ids, documents, metadatas
        )
        used_centroids = True
        info_message = (
            f"⚠️ Sample exceeded {max_pts:,} points for {method}. "
            f"Showing one centroid per label instead."
        )

    # -------------------------------------------------------------------
    # Dimensionality reduction
    # -------------------------------------------------------------------
    if method == "PCA":
        coords = _run_pca(X)
    elif method == "UMAP":
        coords = _run_umap(X, n_neighbors=umap_n_neighbors, min_dist=umap_min_dist)
    elif method == "t-SNE":
        coords = _run_tsne(X, perplexity=tsne_perplexity, max_iter=tsne_max_iter)
    else:
        raise ValueError(f"Unknown reduction method: {method}")

    # -------------------------------------------------------------------
    # Build DataFrame for Plotly
    # -------------------------------------------------------------------
    # Truncate document text for hover (full text can be enormous)
    hover_texts = [doc[:400] + ("…" if len(doc) > 400 else "") for doc in documents]

    df = pd.DataFrame(
        {
            "x": coords[:, 0],
            "y": coords[:, 1],
            "id": ids,
            "text": hover_texts,
            "color": labels_raw,
        }
    )

    # Add any extra metadata columns for hover
    extra_cols = set()
    for meta in metadatas:
        extra_cols.update(meta.keys())
    extra_cols -= {"chunk_index", "doc_id"}  # internal, skip

    for col in sorted(extra_cols):
        df[col] = [m.get(col, "") for m in metadatas]

    hover_cols = ["id", "text"] + [c for c in sorted(extra_cols) if c != color_by]

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    template = "plotly_dark" if dark_mode else "plotly_white"

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="color" if color_by else None,
        hover_data={col: True for col in hover_cols},
        title=f"{method} — {len(ids):,} points"
        + (" (centroids)" if used_centroids else ""),
        labels={
            "x": f"{method} dim 1",
            "y": f"{method} dim 2",
            "color": color_by or "",
        },
        template=template,
        opacity=point_opacity,
    )

    fig.update_traces(marker=dict(size=point_size))
    fig.update_layout(
        legend_title_text=color_by or "",
        hoverlabel=dict(
            bgcolor="rgba(30,30,30,0.85)",
            font_size=12,
            font_family="monospace",
        ),
        margin=dict(l=20, r=20, t=50, b=20),
        height=620,
    )

    return fig, used_centroids, info_message


# ---------------------------------------------------------------------------
# Sampling helper (called by the UI before reduce_and_plot)
# ---------------------------------------------------------------------------


def compute_sample_size(
    total: int,
    fraction: float,
    method: str,
) -> tuple[int, bool, str]:
    """
    Given the collection size and the user's chosen fraction, return:
    (n_to_sample, will_use_centroids, warning_message)
    """
    n = max(1, int(total * fraction))

    max_pts = REDUCTION_MAX_POINTS.get(method)
    will_centroids = False
    warning = ""

    if max_pts and n > max_pts:
        warning = (
            f"{method} is stable up to ~{max_pts:,} points. "
            f"Your selection ({n:,}) exceeds this — centroids per label will be used."
        )
        will_centroids = True

    # Centroid fallback by fraction
    if (
        not will_centroids
        and total >= CENTROID_FALLBACK_MIN_DOCS
        and fraction > CENTROID_FALLBACK_FRACTION
    ):
        warning = (
            f"Sample fraction ({fraction:.0%}) is large for a {total:,}-item collection. "
            f"Centroids per label will be used to keep the visualisation stable."
        )
        will_centroids = True

    return n, will_centroids, warning
