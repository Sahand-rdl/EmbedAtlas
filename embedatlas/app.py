"""
EmbedAtlas — Home / Landing Page
Entry point launched by `streamlit run embedatlas/app.py`
or via the `embedatlas` CLI command.
"""

import streamlit as st

from embedatlas.config import APP_ICON, APP_TITLE, APP_VERSION
from embedatlas.core.vectorstore import VectorStore
from embedatlas.ui.components import collection_status_card
from embedatlas.ui.sidebar import render_sidebar

st.set_page_config(
    page_title=f"{APP_TITLE}",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.markdown(
    f"""
    <div style="text-align:center; padding: 2rem 0 1rem 0;">
        <span style="font-size: 3.5rem;">{APP_ICON}</span>
        <h1 style="font-size: 2.8rem; font-weight: 800; margin: 0.2rem 0;">
            {APP_TITLE}
        </h1>
        <p style="color: gray; font-size: 1.1rem; margin-top: 0.4rem;">
            The visual embedding workbench &nbsp;·&nbsp; v{APP_VERSION}
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.divider()

# ---------------------------------------------------------------------------
# Step cards
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)

cards = [
    (
        "1️⃣",
        "Ingest",
        "Load text from files, folders, HuggingFace datasets, or any URL.",
        "01_ingest",
    ),
    (
        "2️⃣",
        "Embed",
        "Pick a SentenceTransformer model and generate vector embeddings.",
        "02_embed",
    ),
    (
        "3️⃣",
        "Explore",
        "Visualise clusters with interactive PCA, UMAP, or t-SNE via Plotly.",
        "03_explore",
    ),
    (
        "4️⃣",
        "Search",
        "Query your data with keyword, semantic, or hybrid RAG search.",
        "04_search",
    ),
]

for col, (icon, title, desc, _page) in zip([c1, c2, c3, c4], cards):
    with col:
        with st.container(border=True):
            st.markdown(f"### {icon} {title}")
            st.caption(desc)

st.divider()

# ---------------------------------------------------------------------------
# Collections overview
# ---------------------------------------------------------------------------
st.markdown("### Your collections")

vs = VectorStore()
collections = vs.list_collections()

if not collections:
    st.info(
        "No collections yet. Click **1 · Ingest** in the sidebar to create your first one.",
        icon="📭",
    )
else:
    cols = st.columns(min(len(collections), 3))
    for i, info in enumerate(collections):
        with cols[i % 3]:
            collection_status_card(info)

st.divider()

# ---------------------------------------------------------------------------
# Quick-start guide
# ---------------------------------------------------------------------------
with st.expander("📖 Quick-start guide", expanded=not bool(collections)):
    st.markdown(
        """
        **1 · Ingest** — Upload a `.txt`, `.md`, `.pdf`, `.csv`, or `.json` file,
        point to a local folder, paste a HuggingFace dataset ID (e.g. `wikitext`),
        or drop in a raw URL. Give your data a **label** so points are colour-coded
        in the visualiser.

        **2 · Embed** — Choose an embedding model. `all-mpnet-base-v2` is a great
        all-round default. Smaller models like `all-MiniLM-L6-v2` are faster on CPU.
        Multilingual data? Use `BAAI/bge-m3` or `paraphrase-multilingual-MiniLM-L12-v2`.

        **3 · Explore** — Run PCA, UMAP, or t-SNE on your embeddings. Hover any point
        to read the original chunk. Colour by any metadata field you attached during
        ingestion (e.g. language, source, topic).

        **4 · Search** — Ask questions or search for concepts. Hybrid mode surfaces
        exact keyword matches first, then fills with semantically related chunks.
        Export results as CSV or JSON.

        ---
        **Tips**
        - Always attach a **label** during ingestion — it makes visualisations meaningful.
        - For RAG, use the same embedding model for ingestion and search.
        - Large datasets? Set a reasonable **max rows** limit on the HF tab and use
          the sample-size slider in Explore to keep t-SNE stable.
        """
    )

st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px; margin-top:2rem;'>"
    "EmbedAtlas is open source · "
    "<a href='https://github.com/your-org/embedatlas'>GitHub</a>"
    "</div>",
    unsafe_allow_html=True,
)
