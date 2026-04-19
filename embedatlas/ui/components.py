"""
EmbedAtlas — Reusable UI Components
All shared Streamlit widgets live here so pages stay thin.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

import streamlit as st

from embedatlas.config import APP_ICON, APP_TITLE, APP_VERSION
from embedatlas.core.rag import SearchResult
from embedatlas.core.vectorstore import CollectionInfo


# ---------------------------------------------------------------------------
# Page shell
# ---------------------------------------------------------------------------


def page_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.caption(subtitle)
    st.divider()


# ---------------------------------------------------------------------------
# Collection status card  (used in sidebar and top of each page)
# ---------------------------------------------------------------------------


def collection_status_card(info: CollectionInfo) -> None:
    """Small card showing name / chunk count / model."""
    with st.container(border=True):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{info.name}**")
            model_short = (info.model_id or "unknown").split("/")[-1]
            st.caption(f"model: `{model_short}`")
        with col2:
            st.metric("chunks", f"{info.count:,}")


# ---------------------------------------------------------------------------
# Progress helpers
# ---------------------------------------------------------------------------


def make_progress_callback(
    bar: Any,
    status_text: Any,
    label: str = "Processing",
) -> Callable[[int, int], None]:
    """
    Returns a callback(done, total) suitable for Embedder.embed_chunks().
    Wraps a Streamlit progress bar + status text widget.
    """

    def callback(done: int, total: int) -> None:
        pct = done / total if total else 0
        bar.progress(pct)
        status_text.caption(f"{label}: {done:,} / {total:,}")

    return callback


def simple_progress_callback(
    bar: Any,
    status_text: Any,
    label: str = "Ingesting rows",
) -> Callable[[int], None]:
    """
    Returns a callback(current_row) for ingest_hf_dataset's progress_callback.
    No 'total' known in advance.
    """
    start = time.time()

    def callback(current: int) -> None:
        elapsed = time.time() - start
        bar.progress(min(current / 10_000, 1.0))  # rough visual
        status_text.caption(f"{label}: {current:,} rows — {elapsed:.1f}s elapsed")

    return callback


# ---------------------------------------------------------------------------
# Search result display
# ---------------------------------------------------------------------------

MATCH_BADGE_COLORS = {
    "both": "🟢",
    "keyword": "🔵",
    "semantic": "🟣",
}


def render_search_results(results: List[SearchResult], query: str = "") -> None:
    """Render a list of SearchResults as expandable cards."""
    if not results:
        st.info("No results found. Try a different query or search mode.")
        return

    st.caption(f"{len(results)} result(s)" + (f" for **{query}**" if query else ""))

    for rank, r in enumerate(results, start=1):
        badge = MATCH_BADGE_COLORS.get(r.match_type, "⚪")
        pct = f"{r.score * 100:.1f}%"
        header = f"{badge} **#{rank}** — relevance {pct}  `{r.match_type}`"

        with st.expander(header, expanded=(rank <= 3)):
            st.markdown(r.document)
            if r.metadata:
                st.divider()
                meta_cols = st.columns(min(len(r.metadata), 4))
                for i, (k, v) in enumerate(r.metadata.items()):
                    meta_cols[i % len(meta_cols)].caption(f"**{k}**: {v}")
            st.caption(f"chunk id: `{r.id}`")


# ---------------------------------------------------------------------------
# Chunk-size / overlap slider pair
# ---------------------------------------------------------------------------


def chunking_controls(
    default_size: int = 1000,
    default_overlap: int = 150,
) -> tuple[int, int]:
    """Render chunk size + overlap sliders. Returns (chunk_size, overlap)."""
    col1, col2 = st.columns(2)
    with col1:
        chunk_size = st.slider(
            "Chunk size (characters)",
            min_value=100,
            max_value=4000,
            value=default_size,
            step=50,
            help="Target length of each text chunk. Smaller = more precise retrieval.",
        )
    with col2:
        overlap = st.slider(
            "Chunk overlap (characters)",
            min_value=0,
            max_value=chunk_size // 2,
            value=min(default_overlap, chunk_size // 2),
            step=25,
            help="Characters shared between consecutive chunks. Prevents context loss at boundaries.",
        )
    return chunk_size, overlap


# ---------------------------------------------------------------------------
# Sample-size slider  (Photoshop-style: slide + exact number entry)
# ---------------------------------------------------------------------------


def sample_size_control(
    total: int,
    method: str,
    default_fraction: float = 0.05,
) -> tuple[int, float]:
    """
    Renders a slider + number_input pair (kept in sync via session state).
    Returns (n_samples, fraction).
    """
    from embedatlas.config import REDUCTION_MAX_POINTS

    max_pts = REDUCTION_MAX_POINTS.get(method) or total
    max_pts = min(max_pts, total)
    default_n = max(1, int(total * default_fraction))

    key_frac = f"sample_frac_{method}"
    key_n = f"sample_n_{method}"

    if key_n not in st.session_state:
        st.session_state[key_n] = default_n
        st.session_state[key_frac] = default_fraction

    col1, col2 = st.columns([3, 1])

    with col1:
        pct = st.slider(
            "Sample size",
            min_value=1,
            max_value=max_pts,
            value=st.session_state[key_n],
            step=max(1, max_pts // 200),
            format="%d pts",
            key=f"slider_{method}",
        )
        st.session_state[key_n] = pct
        st.session_state[key_frac] = pct / total

    with col2:
        exact = st.number_input(
            "Exact",
            min_value=1,
            max_value=max_pts,
            value=st.session_state[key_n],
            step=1,
            key=f"exact_{method}",
            label_visibility="visible",
        )
        if exact != st.session_state[key_n]:
            st.session_state[key_n] = exact
            st.session_state[key_frac] = exact / total
            st.rerun()

    st.caption(
        f"{st.session_state[key_n]:,} of {total:,} points "
        f"({st.session_state[key_frac]:.1%})"
    )

    return st.session_state[key_n], st.session_state[key_frac]


# ---------------------------------------------------------------------------
# Metadata label assignment widget  (used in ingest page)
# ---------------------------------------------------------------------------


def label_input(
    key: str = "label_input",
    placeholder: str = "e.g.  english,  arxiv,  product-reviews",
) -> Optional[str]:
    """
    Single text input for the user-defined label/category.
    Returns the stripped string or None if empty.
    """
    val = st.text_input(
        "Label / Category  *(optional but recommended for visualisation)*",
        placeholder=placeholder,
        key=key,
        help=(
            "Attach a category label to every chunk in this ingestion batch. "
            "Labels are used to colour data-points in the visualisation explorer."
        ),
    )
    return val.strip() or None


# ---------------------------------------------------------------------------
# Warning / info banners
# ---------------------------------------------------------------------------


def warn_large_dataset(n_rows: int, threshold: int = 50_000) -> None:
    if n_rows > threshold:
        st.warning(
            f"⚠️ You selected **{n_rows:,} rows**. "
            f"Embedding this many chunks may take a long time and require significant RAM/VRAM. "
            f"Consider reducing the row count, or use a cloud embedding API for very large datasets.",
            icon="⚠️",
        )


def warn_oom_risk(batch_size: int) -> None:
    if batch_size > 128:
        st.info(
            "💡 Large batch sizes can cause out-of-memory errors on CPU or low-VRAM GPUs. "
            "If embedding crashes, reduce the batch size below.",
            icon="💡",
        )


def info_centroid_mode(method: str) -> None:
    st.info(
        f"ℹ️ Your sample is too large for {method} to remain stable. "
        f"EmbedAtlas will display one **centroid** per label instead of individual points. "
        f"Reduce the sample size slider to see individual points.",
        icon="ℹ️",
    )
