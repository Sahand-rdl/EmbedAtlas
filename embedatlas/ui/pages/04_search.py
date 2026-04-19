"""
EmbedAtlas — Step 4: Search
Keyword / Semantic / Hybrid search over an embedded collection.
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import DEFAULT_TOP_K, EMBEDDING_MODELS, MAX_TOP_K
from embedatlas.core.exporter import results_to_csv, results_to_json
from embedatlas.core.rag import RAGEngine
from embedatlas.core.vectorstore import VectorStore
from embedatlas.ui.components import page_header, render_search_results
from embedatlas.ui.sidebar import ACTIVE_COLLECTION_KEY, render_sidebar

st.set_page_config(page_title="Search · EmbedAtlas", layout="wide")
render_sidebar()

page_header(
    "4️⃣  Search",
    "Query your collection with keyword, semantic, or hybrid search.",
)

vs = VectorStore()
active = st.session_state.get(ACTIVE_COLLECTION_KEY)

if not active:
    st.warning("No active collection. Select one in the sidebar.")
    st.stop()

info = vs.get_collection_info(active)
if info.count == 0:
    st.warning("This collection has no embeddings. Complete **2 · Embed** first.")
    st.stop()

st.markdown(f"**Collection:** `{active}` — **{info.count:,}** chunks")
st.divider()

# ---------------------------------------------------------------------------
# Search config
# ---------------------------------------------------------------------------
cfg_col, res_col = st.columns([1, 2])

with cfg_col:
    st.markdown("### Search mode")
    mode = st.radio(
        "Mode",
        ["Hybrid", "Semantic", "Keyword"],
        index=0,
        label_visibility="collapsed",
        help=(
            "**Hybrid**: keyword hits ranked first, semantic results fill the rest.\n\n"
            "**Semantic**: pure vector similarity — finds conceptually related chunks "
            "even without exact word matches.\n\n"
            "**Keyword**: BM25 exact/fuzzy term matching across all stored text."
        ),
    )

    st.markdown("### Model")
    # Infer model from collection metadata if available
    stored_model = info.metadata.get("model_id")
    model_ids = [m["model_id"] for m in EMBEDDING_MODELS]

    if stored_model and stored_model in model_ids:
        model_display = next(
            m["display_name"] for m in EMBEDDING_MODELS if m["model_id"] == stored_model
        )
        st.info(f"Using collection model: `{stored_model.split('/')[-1]}`", icon="🔗")
        selected_model_id = stored_model
    else:
        display_names = [m["display_name"] for m in EMBEDDING_MODELS]
        selected_display = st.selectbox(
            "Embedding model",
            options=display_names,
            index=0,
            label_visibility="collapsed",
            help="Must match the model used during embedding, otherwise results will be poor.",
        )
        selected_model_id = next(
            m["model_id"]
            for m in EMBEDDING_MODELS
            if m["display_name"] == selected_display
        )

    st.markdown("### Results")
    top_k = st.slider(
        "Top-K results",
        min_value=1,
        max_value=MAX_TOP_K,
        value=DEFAULT_TOP_K,
    )

    # Optional metadata filter
    st.markdown("### Filter  *(optional)*")
    if info.metadata_keys:
        filter_key = st.selectbox(
            "Filter by field",
            options=["(no filter)"] + info.metadata_keys,
        )
        if filter_key != "(no filter)":
            unique_vals = vs.get_unique_metadata_values(active, filter_key)
            filter_val = st.selectbox("Value", options=unique_vals)
            where_filter = {filter_key: filter_val}
        else:
            where_filter = None
    else:
        st.caption("No metadata fields found in this collection.")
        where_filter = None

# ---------------------------------------------------------------------------
# Engine (cached per collection + model to avoid reloading BM25)
# ---------------------------------------------------------------------------
engine_key = f"rag_engine_{active}_{selected_model_id}"
if engine_key not in st.session_state:
    with st.spinner("Initialising search engine…"):
        try:
            st.session_state[engine_key] = RAGEngine(
                collection_name=active,
                model_id=selected_model_id,
                top_k=top_k,
            )
        except Exception as e:
            st.error(f"Failed to initialise search engine: {e}")
            st.stop()

engine: RAGEngine = st.session_state[engine_key]

# ---------------------------------------------------------------------------
# Query input + search
# ---------------------------------------------------------------------------
with res_col:
    query = st.text_input(
        "Search query",
        placeholder="What is the capital of Germany?",
        label_visibility="collapsed",
    ).strip()

    if not query:
        st.info("Enter a query above to search.", icon="🔍")
        st.stop()

    with st.spinner(f"Running {mode.lower()} search…"):
        try:
            results = engine.search(
                query=query,
                mode=mode.lower(),
                top_k=top_k,
                where=where_filter if mode.lower() != "keyword" else None,
            )
        except Exception as e:
            st.error(f"Search failed: {e}")
            st.stop()

    render_search_results(results, query=query)

    if results:
        st.divider()
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️  Export as CSV",
                data=results_to_csv(results),
                file_name=f"{active}_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️  Export as JSON",
                data=results_to_json(results),
                file_name=f"{active}_results.json",
                mime="application/json",
                use_container_width=True,
            )
