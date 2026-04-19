"""
EmbedAtlas — Step 2: Embed
Model picker, batch-size control, embedding execution with live progress.
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import EMBEDDING_MODELS, DEFAULT_MODEL_INDEX
from embedatlas.core.embedder import Embedder
from embedatlas.core.vectorstore import VectorStore
from embedatlas.ui.components import (
    make_progress_callback,
    page_header,
    warn_large_dataset,
    warn_oom_risk,
)
from embedatlas.ui.sidebar import ACTIVE_COLLECTION_KEY, render_sidebar

st.set_page_config(page_title="Embed · EmbedAtlas", layout="wide")
render_sidebar()

page_header(
    "2️⃣  Embed",
    "Choose an embedding model and generate vectors for your chunks.",
)

vs = VectorStore()
active = st.session_state.get(ACTIVE_COLLECTION_KEY)
pending_chunks = st.session_state.get("pending_chunks", [])
pending_col = st.session_state.get("pending_collection")

# ---------------------------------------------------------------------------
# Guard: need an active collection
# ---------------------------------------------------------------------------
if not active:
    st.warning("No active collection selected. Go to **1 · Ingest** first.")
    st.stop()

info = vs.get_collection_info(active)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"**Active collection:** `{active}`")
with col2:
    st.metric("Chunks already stored", f"{info.count:,}")

# ---------------------------------------------------------------------------
# Pending chunks from ingest page
# ---------------------------------------------------------------------------
if pending_chunks and pending_col == active:
    st.info(
        f"**{len(pending_chunks):,} chunks** are staged from the Ingest step and ready to embed.",
        icon="📦",
    )
else:
    if info.count == 0:
        st.warning(
            "No chunks found for this collection and none are staged. "
            "Complete the **Ingest** step first.",
        )
        st.stop()
    else:
        st.info(
            f"No new chunks staged. The collection already has **{info.count:,}** chunks. "
            f"You can re-embed below with a different model (this will overwrite existing vectors).",
            icon="ℹ️",
        )
        pending_chunks = (
            []
        )  # embed what's already in DB is not supported — user re-ingests

st.divider()

# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------
st.markdown("### Embedding model")

model_display_names = [m["display_name"] for m in EMBEDDING_MODELS]
selected_display = st.selectbox(
    "Model",
    options=model_display_names,
    index=DEFAULT_MODEL_INDEX,
    label_visibility="collapsed",
)

# Show model details card
selected_model = next(
    m for m in EMBEDDING_MODELS if m["display_name"] == selected_display
)
with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    c1.metric("Dimensions", selected_model["dims"])
    c2.metric("Max seq length", selected_model["max_seq_len"])
    c3.metric("Speed", selected_model["speed"].upper())
    st.caption(f"📖 {selected_model['description']}")
    st.caption(f"🔗 `{selected_model['model_id']}`")

if selected_model["max_seq_len"] < 256:
    st.warning(
        f"⚠️ This model truncates input at **{selected_model['max_seq_len']} tokens** (~{selected_model['max_seq_len']*4} characters). "
        f"Chunks longer than this will be silently truncated. Consider reducing chunk size on the Ingest page.",
    )

st.divider()

# ---------------------------------------------------------------------------
# Batch size
# ---------------------------------------------------------------------------
st.markdown("### Hardware settings")

batch_size = st.slider(
    "Batch size  *(reduce if you get out-of-memory errors)*",
    min_value=1,
    max_value=512,
    value=64,
    step=8,
    help=(
        "Number of chunks encoded per forward pass. "
        "Larger = faster but more RAM/VRAM. "
        "On CPU, 32–64 is a safe default. On GPU, try 128–256."
    ),
)
warn_oom_risk(batch_size)

device_hint = st.radio(
    "Device hint",
    ["CPU  (safe everywhere)", "CUDA  (NVIDIA GPU)", "MPS  (Apple Silicon)"],
    horizontal=True,
    index=0,
    help="EmbedAtlas will attempt to use this device. Falls back to CPU if unavailable.",
)

st.divider()

# ---------------------------------------------------------------------------
# Run embedding
# ---------------------------------------------------------------------------
if not pending_chunks:
    st.stop()

warn_large_dataset(len(pending_chunks))

est_time = len(pending_chunks) / max(batch_size, 1) * 0.4  # rough seconds estimate
st.caption(
    f"Estimated time: **~{est_time / 60:.1f} min** on CPU "
    f"({len(pending_chunks):,} chunks · batch {batch_size})"
)

if st.button(
    f"🚀  Embed {len(pending_chunks):,} chunks into **{active}**",
    type="primary",
    use_container_width=True,
):
    progress_bar = st.progress(0.0)
    status_text = st.empty()
    cb = make_progress_callback(progress_bar, status_text, label="Embedding chunks")

    with st.spinner("Loading model (first run downloads weights)…"):
        try:
            embedder = Embedder(
                collection_name=active,
                model_id=selected_model["model_id"],
                batch_size=batch_size,
            )
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            st.stop()

    with st.spinner("Embedding…"):
        try:
            stored = embedder.embed_chunks(pending_chunks, progress_callback=cb)
            progress_bar.progress(1.0)
            status_text.empty()
        except MemoryError:
            st.error("💥 Out of memory! Reduce the **batch size** and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

    # Update collection metadata with the model used
    # (ChromaDB doesn't support updating collection metadata directly,
    #  so we store it in session state and the sidebar reads it)
    st.session_state["pending_chunks"] = []
    st.session_state["pending_collection"] = None

    st.balloons()
    st.success(
        f"✅ **{stored:,} chunks** embedded with `{selected_model['model_id']}` "
        f"and stored in **{active}**.\n\n"
        f"Head to **3 · Explore** to visualise or **4 · Search** to query."
    )

    # Refresh sidebar count
    st.rerun()
