"""
EmbedAtlas — Step 2: Embed
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

if not active:
    st.warning("No active collection selected. Go to **1 · Ingest** first.")
    st.stop()

# Persistent success banner — survives rerun after embedding
if st.session_state.get(f"embed_success_{active}"):
    model_short = st.session_state.get(f"model_id_{active}", "").split("/")[-1]
    n_stored = st.session_state.get(f"embed_count_{active}", 0)
    st.success(
        f"✅  **{n_stored:,} chunks** successfully embedded with `{model_short}`.  Use **Explore** or **Search** to continue."
    )

info = vs.get_collection_info(active)

col1, col2 = st.columns([2, 1])
with col1:
    st.markdown(f"**Active collection:** `{active}`")
with col2:
    st.metric("Chunks stored", f"{info.count:,}")

if pending_chunks and pending_col == active:
    st.info(
        f"**{len(pending_chunks):,} chunks** staged from Ingest and ready to embed.",
        icon="📦",
    )
elif info.count == 0:
    st.warning("No chunks staged and none in DB. Complete **1 · Ingest** first.")
    st.stop()
else:
    st.info(
        f"No new chunks staged. Collection already has **{info.count:,}** chunks. Re-ingest to embed with a different model.",
        icon="ℹ️",
    )
    pending_chunks = []

st.divider()

st.markdown("### Embedding model")

model_display_names = [m["display_name"] for m in EMBEDDING_MODELS]
selected_display = st.selectbox(
    "Model",
    options=model_display_names,
    index=DEFAULT_MODEL_INDEX,
    label_visibility="collapsed",
)
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
        f"⚠️ This model truncates at **{selected_model['max_seq_len']} tokens**. Consider reducing chunk size on the Ingest page."
    )

st.divider()

st.markdown("### Hardware settings")

batch_size = st.slider(
    "Batch size  *(reduce if you get out-of-memory errors)*",
    min_value=1,
    max_value=512,
    value=64,
    step=8,
    help="Chunks per forward pass. CPU: 32-64. GPU: 128-256.",
)
warn_oom_risk(batch_size)

st.radio(
    "Device hint",
    ["CPU  (safe everywhere)", "CUDA  (NVIDIA GPU)", "MPS  (Apple Silicon)"],
    horizontal=True,
    index=0,
    help="SentenceTransformers auto-detects your GPU. This is informational only.",
)

st.divider()

if not pending_chunks:
    st.stop()

warn_large_dataset(len(pending_chunks))

est_min = len(pending_chunks) / max(batch_size, 1) * 0.4 / 60
st.caption(
    f"Estimated time: **~{est_min:.1f} min** on CPU ({len(pending_chunks):,} chunks · batch {batch_size})"
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
            st.error("💥 Out of memory! Reduce batch size and try again.")
            st.stop()
        except Exception as e:
            st.error(f"Embedding failed: {e}")
            st.stop()

    st.session_state[f"model_id_{active}"] = selected_model["model_id"]
    st.session_state[f"embed_success_{active}"] = True
    st.session_state[f"embed_count_{active}"] = stored
    st.session_state["pending_chunks"] = []
    st.session_state["pending_collection"] = None
    st.rerun()
