"""
EmbedAtlas — Step 3: Explore
PCA / UMAP / t-SNE interactive Plotly scatter plots.
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import DEFAULT_SAMPLE_FRACTION, TSNE_DEFAULTS, UMAP_DEFAULTS
from embedatlas.core.exporter import sample_to_csv, sample_to_json
from embedatlas.core.reduction import compute_sample_size, reduce_and_plot
from embedatlas.core.vectorstore import VectorStore
from embedatlas.ui.components import (
    info_centroid_mode,
    page_header,
    sample_size_control,
)
from embedatlas.ui.sidebar import ACTIVE_COLLECTION_KEY, render_sidebar

st.set_page_config(page_title="Explore · EmbedAtlas", layout="wide")
render_sidebar()

page_header(
    "3️⃣  Explore",
    "Visualise your embeddings in 2D using PCA, UMAP, or t-SNE.",
)

vs = VectorStore()
active = st.session_state.get(ACTIVE_COLLECTION_KEY)

if not active:
    st.warning("No active collection. Select one in the sidebar.")
    st.stop()

info = vs.get_collection_info(active)

if info.count == 0:
    st.warning("This collection has no embeddings yet. Complete **2 · Embed** first.")
    st.stop()

st.markdown(f"**Collection:** `{active}` — **{info.count:,}** chunks")
st.divider()

# ── Layout: controls left, plot right ─────────────────────────────────────
ctrl_col, plot_col = st.columns([1, 3])

with ctrl_col:
    st.markdown("### Method")
    method = st.radio(
        "Reduction method",
        ["PCA", "UMAP", "t-SNE"],
        index=1,
        label_visibility="collapsed",
    )

    st.markdown("### Colour by")
    color_options = ["(none)"] + info.metadata_keys
    color_by_raw = st.selectbox(
        "Metadata field",
        options=color_options,
        index=0,
        label_visibility="collapsed",
        help="Pick a metadata field to colour points. Set a label during Ingest for best results.",
    )
    color_by = None if color_by_raw == "(none)" else color_by_raw

    st.markdown("### Sample size")
    n_samples, fraction = sample_size_control(
        total=info.count,
        method=method,
        default_fraction=DEFAULT_SAMPLE_FRACTION,
    )

    st.markdown("### Parameters")
    if method == "t-SNE":
        perplexity = st.slider(
            "Perplexity",
            min_value=2,
            max_value=200,
            value=TSNE_DEFAULTS["perplexity"],
            help="Balance local vs global structure. Typical: 5–50.",
        )
        max_iter = st.slider(
            "Max iterations",
            min_value=100,
            max_value=2000,
            value=TSNE_DEFAULTS["max_iter"],
            step=100,
        )
    elif method == "UMAP":
        n_neighbors = st.slider(
            "n_neighbors",
            min_value=2,
            max_value=200,
            value=UMAP_DEFAULTS["n_neighbors"],
            help="Neighbourhood size. Smaller = more local detail.",
        )
        min_dist = st.slider(
            "min_dist",
            min_value=0.0,
            max_value=1.0,
            value=UMAP_DEFAULTS["min_dist"],
            step=0.05,
            help="Minimum distance between points. Smaller = tighter clusters.",
        )

    with st.expander("Visual settings"):
        point_size = st.slider("Point size", 2, 20, 6)
        point_opacity = st.slider("Opacity", 0.1, 1.0, 0.75, step=0.05)
        dark_mode = st.toggle("Dark background", value=False)

    st.divider()
    run_btn = st.button(
        "▶  Run visualisation", type="primary", use_container_width=True
    )

# ── Guard: don't enter plot_col with st.stop() — render placeholder instead
if not run_btn:
    with plot_col:
        st.info(
            "Configure the options on the left and click **Run visualisation**.",
            icon="👈",
        )
else:
    with plot_col:
        # Centroid warning
        _, will_centroids, _ = compute_sample_size(info.count, fraction, method)
        if will_centroids:
            info_centroid_mode(method)

        # Sample from ChromaDB
        with st.spinner(f"Sampling {n_samples:,} chunks from ChromaDB…"):
            try:
                sample = vs.sample_embeddings(active, n=n_samples)
            except Exception as e:
                st.error(f"Failed to load embeddings: {e}")
                st.stop()

        if not sample["ids"]:
            st.error("No embeddings returned. Try re-embedding the collection.")
            st.stop()

        # Build kwargs
        kwargs: dict = dict(
            sample=sample,
            method=method,
            color_by=color_by,
            point_size=point_size,
            point_opacity=point_opacity,
            dark_mode=dark_mode,
        )
        if method == "t-SNE":
            kwargs["tsne_perplexity"] = perplexity
            kwargs["tsne_max_iter"] = max_iter
        elif method == "UMAP":
            kwargs["umap_n_neighbors"] = n_neighbors
            kwargs["umap_min_dist"] = min_dist

        # Run reduction
        with st.spinner(f"Running {method} on {len(sample['ids']):,} points…"):
            try:
                fig, used_centroids, info_msg = reduce_and_plot(**kwargs)
            except Exception as e:
                st.error(f"Reduction failed: {e}")
                st.stop()

        if info_msg:
            st.info(info_msg, icon="ℹ️")

        st.plotly_chart(fig, use_container_width=True)

        exp1, exp2 = st.columns(2)
        with exp1:
            st.download_button(
                "⬇️  Export sample as CSV",
                data=sample_to_csv(sample),
                file_name=f"{active}_sample.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with exp2:
            st.download_button(
                "⬇️  Export sample as JSON",
                data=sample_to_json(sample, include_embeddings=False),
                file_name=f"{active}_sample.json",
                mime="application/json",
                use_container_width=True,
            )
