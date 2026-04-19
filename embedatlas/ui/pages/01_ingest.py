"""
EmbedAtlas — Step 1: Ingest
Handles local files, local folders, HuggingFace datasets, and raw URLs.
Produces chunked documents and stores them ready for embedding.
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_PDF_PARSER,
    HF_DEFAULT_MAX_ROWS,
    HF_DEFAULT_TEXT_COLUMN,
    HF_MAX_ROWS_HARD_CAP,
    SUPPORTED_EXTENSIONS,
    TEMP_DIR,
)
from embedatlas.core.chunker import Chunker
from embedatlas.core.ingestion import (
    ingest_hf_dataset,
    ingest_local_file,
    ingest_local_folder,
    ingest_url,
)
from embedatlas.core.vectorstore import VectorStore
from embedatlas.ui.components import (
    chunking_controls,
    label_input,
    page_header,
    simple_progress_callback,
    warn_large_dataset,
)
from embedatlas.ui.sidebar import ACTIVE_COLLECTION_KEY, render_sidebar

st.set_page_config(page_title="Ingest · EmbedAtlas", layout="wide")
render_sidebar()

page_header(
    "1️⃣  Ingest",
    "Load your dataset from a local file, folder, HuggingFace, or a URL.",
)

vs = VectorStore()

# ---------------------------------------------------------------------------
# Collection target
# ---------------------------------------------------------------------------
st.markdown("### Target collection")

col_mode = st.radio(
    "Collection",
    ["Create new", "Add to existing"],
    horizontal=True,
)

if col_mode == "Create new":
    collection_name = st.text_input(
        "New collection name",
        placeholder="my-rag-collection",
    ).strip()
else:
    existing = vs.collection_names()
    if not existing:
        st.warning("No existing collections. Create one first.")
        st.stop()
    collection_name = st.selectbox("Existing collection", existing)

if not collection_name:
    st.info("Enter a collection name to continue.")
    st.stop()

st.divider()

# ---------------------------------------------------------------------------
# Source selection
# ---------------------------------------------------------------------------
st.markdown("### Data source")

source_tab = st.tabs(
    ["📁 Local file(s)", "📂 Local folder", "🤗 HuggingFace dataset", "🔗 URL"]
)

documents = []  # list of {"text", "doc_id", "metadata"} dicts
source_ready = False

# ── Tab 1: Local files ──────────────────────────────────────────────────────
with source_tab[0]:
    all_exts = [e for exts in SUPPORTED_EXTENSIONS.values() for e in exts]
    uploaded = st.file_uploader(
        "Upload one or more files",
        type=[e.lstrip(".") for e in all_exts],
        accept_multiple_files=True,
    )

    pdf_parser = st.radio(
        "PDF parser",
        [
            "pymupdf  (fast, default)",
            "docling  (layout-aware, requires separate install)",
        ],
        horizontal=True,
        index=0,
    )
    pdf_parser_key = "pymupdf" if "pymupdf" in pdf_parser else "docling"

    label = label_input(key="label_local_files")
    text_col = (
        st.text_input(
            "Text column  *(CSV / JSON only)*",
            placeholder="text",
            help="Column name that contains the main text. Leave blank to use all columns.",
        ).strip()
        or None
    )

    if uploaded:
        source_ready = True
        if st.button("📥 Load files", type="primary", key="btn_load_files"):
            TEMP_DIR.mkdir(parents=True, exist_ok=True)
            with st.spinner("Reading files…"):
                for f in uploaded:
                    tmp = TEMP_DIR / f.name
                    tmp.write_bytes(f.read())
                    try:
                        docs = ingest_local_file(
                            tmp,
                            label=label,
                            pdf_parser=pdf_parser_key,
                            text_column=text_col,
                        )
                        documents.extend(docs)
                    except Exception as e:
                        st.error(f"{f.name}: {e}")
                    finally:
                        tmp.unlink(missing_ok=True)

            st.success(f"Loaded {len(documents)} document(s).")
            st.session_state["staged_documents"] = documents

# ── Tab 2: Local folder ─────────────────────────────────────────────────────
with source_tab[1]:
    st.info(
        "Enter the **absolute path** to a folder on this machine. "
        "All supported files inside will be loaded recursively.",
        icon="ℹ️",
    )
    folder_path = st.text_input("Folder path", placeholder="/Users/you/my-documents/")
    label_f = label_input(key="label_folder")
    recursive = st.checkbox("Recursive (include sub-folders)", value=True)

    if folder_path:
        if st.button("📥 Load folder", type="primary", key="btn_load_folder"):
            with st.spinner("Reading folder…"):
                try:
                    documents = ingest_local_folder(
                        folder_path,
                        label=label_f,
                        recursive=recursive,
                    )
                    st.success(f"Loaded {len(documents)} document(s).")
                    st.session_state["staged_documents"] = documents
                except Exception as e:
                    st.error(str(e))

# ── Tab 3: HuggingFace ──────────────────────────────────────────────────────
with source_tab[2]:
    repo_id = st.text_input(
        "HuggingFace dataset repo",
        placeholder="allenai/c4",
        help="The dataset repository ID from huggingface.co/datasets",
    ).strip()

    hf_col1, hf_col2 = st.columns(2)
    with hf_col1:
        split = st.text_input("Split", value="train")
    with hf_col2:
        text_col_hf = st.text_input("Text column", value=HF_DEFAULT_TEXT_COLUMN)

    label_col_hf = (
        st.text_input(
            "Label column  *(optional)*",
            placeholder="language",
            help="Column to use as the category label for each row (used for coloring in visualisations).",
        ).strip()
        or None
    )

    max_rows = st.number_input(
        "Max rows to stream",
        min_value=100,
        max_value=HF_MAX_ROWS_HARD_CAP,
        value=HF_DEFAULT_MAX_ROWS,
        step=1000,
        help=f"Hard cap: {HF_MAX_ROWS_HARD_CAP:,}. Streaming stops here regardless of dataset size.",
    )
    warn_large_dataset(max_rows)

    if repo_id:
        if st.button("📥 Stream dataset", type="primary", key="btn_load_hf"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            cb = simple_progress_callback(
                progress_bar, status_text, label="Streaming rows"
            )

            with st.spinner(f"Streaming from {repo_id}…"):
                try:
                    documents = ingest_hf_dataset(
                        repo_id=repo_id,
                        split=split,
                        text_column=text_col_hf,
                        label_column=label_col_hf,
                        max_rows=int(max_rows),
                        progress_callback=cb,
                    )
                    progress_bar.progress(1.0)
                    st.success(f"Streamed {len(documents):,} rows.")
                    st.session_state["staged_documents"] = documents
                except Exception as e:
                    st.error(str(e))

# ── Tab 4: URL ──────────────────────────────────────────────────────────────
with source_tab[3]:
    url = st.text_input(
        "File URL",
        placeholder="https://raw.githubusercontent.com/user/repo/main/data.txt",
        help="Direct link to a .txt .md .csv .json .jsonl or .pdf file.",
    ).strip()

    label_u = label_input(key="label_url")
    text_col_u = (
        st.text_input(
            "Text column  *(CSV / JSON only)*",
            placeholder="text",
            key="url_text_col",
        ).strip()
        or None
    )

    if url:
        if st.button("📥 Download & load", type="primary", key="btn_load_url"):
            with st.spinner(f"Downloading {url}…"):
                try:
                    documents = ingest_url(
                        url,
                        label=label_u,
                        text_column=text_col_u,
                    )
                    st.success(f"Loaded {len(documents)} document(s).")
                    st.session_state["staged_documents"] = documents
                except Exception as e:
                    st.error(str(e))

st.divider()

# ---------------------------------------------------------------------------
# Chunking config + commit
# ---------------------------------------------------------------------------
staged = st.session_state.get("staged_documents", [])

if staged:
    st.markdown(f"### ✅ {len(staged)} document(s) staged")
    with st.expander("Preview first document"):
        first = staged[0]
        st.caption(f"doc_id: `{first['doc_id']}`  |  metadata: `{first['metadata']}`")
        st.text(first["text"][:800] + ("…" if len(first["text"]) > 800 else ""))

    st.markdown("### Chunking settings")
    chunk_size, overlap = chunking_controls(DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP)

    st.divider()
    st.markdown("### Commit to collection")

    if st.button(
        f"✂️  Chunk & save to  **{collection_name}**",
        type="primary",
        use_container_width=True,
    ):
        chunker = Chunker(chunk_size=chunk_size, chunk_overlap=overlap)

        with st.spinner("Chunking documents…"):
            all_chunks = chunker.split_many(staged)

        st.info(
            f"Created **{len(all_chunks):,}** chunks from {len(staged)} document(s)."
        )

        # Persist raw chunks as plain dicts in session for the embed page
        # (the Embedder on page 2 will actually write to ChromaDB)
        st.session_state["pending_chunks"] = all_chunks
        st.session_state["pending_collection"] = collection_name
        st.session_state["pending_chunk_size"] = chunk_size
        st.session_state["pending_overlap"] = overlap

        # Ensure the collection exists in ChromaDB (empty, model assigned on embed)
        if col_mode == "Create new" and not vs.collection_exists(collection_name):
            vs.create_collection(collection_name)

        st.session_state[ACTIVE_COLLECTION_KEY] = collection_name
        st.session_state["staged_documents"] = []

        st.success(
            f"**{len(all_chunks):,} chunks** are ready. "
            f"Head to **2 · Embed** to generate embeddings."
        )
