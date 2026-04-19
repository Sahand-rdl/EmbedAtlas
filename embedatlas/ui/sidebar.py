"""
EmbedAtlas — Sidebar
Persistent collection manager shown on every page.
Manages st.session_state["active_collection"] which all pages read from.
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import APP_ICON, APP_TITLE, APP_VERSION
from embedatlas.core.vectorstore import VectorStore


# Session-state keys used across the whole app
ACTIVE_COLLECTION_KEY = "active_collection"
VS_KEY = "vectorstore"  # shared VectorStore instance


def _get_vs() -> VectorStore:
    """Return (or create) the shared VectorStore instance."""
    if VS_KEY not in st.session_state:
        st.session_state[VS_KEY] = VectorStore()
    return st.session_state[VS_KEY]


def render_sidebar() -> None:
    """
    Call this at the top of every page.
    Renders the full sidebar and sets st.session_state[ACTIVE_COLLECTION_KEY].
    """
    vs = _get_vs()

    with st.sidebar:
        # ── Brand ──────────────────────────────────────────────────────
        st.markdown(f"# {APP_ICON} {APP_TITLE}")
        st.caption(f"v{APP_VERSION}")
        st.divider()

        # ── Collection selector ────────────────────────────────────────
        st.markdown("### Collections")

        collections = vs.list_collections()
        names = [c.name for c in collections]

        if not names:
            st.info("No collections yet.\nHead to **Ingest** to create one.")
            st.session_state[ACTIVE_COLLECTION_KEY] = None
        else:
            # Preserve previously selected collection across reruns
            prev = st.session_state.get(ACTIVE_COLLECTION_KEY)
            default_idx = names.index(prev) if prev in names else 0

            selected = st.selectbox(
                "Active collection",
                options=names,
                index=default_idx,
                label_visibility="collapsed",
            )
            st.session_state[ACTIVE_COLLECTION_KEY] = selected

            # Show info for active collection
            info = next(c for c in collections if c.name == selected)
            _render_collection_card(info)

        st.divider()

        # ── Collection management ──────────────────────────────────────
        with st.expander("⚙️  Manage collections", expanded=False):
            _render_delete_section(vs, names)
            st.divider()
            _render_rename_section(vs, names)

        st.divider()

        # ── Navigation hints ───────────────────────────────────────────
        active = st.session_state.get(ACTIVE_COLLECTION_KEY)
        _render_step_status(active, collections)

        # ── Footer ────────────────────────────────────────────────────
        st.markdown(
            "<br><br><div style='font-size:11px; color:gray;'>"
            "EmbedAtlas · open source<br>"
            "<a href='https://github.com/your-org/embedatlas'>GitHub</a>"
            "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _render_collection_card(info) -> None:
    model_short = (info.model_id or "not embedded yet").split("/")[-1]
    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 6px;
        ">
            <span style="font-size:13px; font-weight:600;">{info.name}</span><br>
            <span style="font-size:11px; color:gray;">
                {info.count:,} chunks &nbsp;·&nbsp; {model_short}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metadata key chips
    if info.metadata_keys:
        st.caption("Metadata fields: " + "  `" + "`  `".join(info.metadata_keys) + "`")


def _render_step_status(active: str | None, collections: list) -> None:
    """Show which steps are available given the active collection."""
    has_collection = active is not None
    info = next((c for c in collections if c.name == active), None)
    has_embeddings = info is not None and info.count > 0

    def step(icon: str, label: str, enabled: bool) -> None:
        color = "inherit" if enabled else "gray"
        prefix = icon if enabled else "🔒"
        st.markdown(
            f"<span style='color:{color}; font-size:13px;'>{prefix} {label}</span>",
            unsafe_allow_html=True,
        )

    st.markdown("**Steps**")
    step("1️⃣", "Ingest", True)  # always available
    step("2️⃣", "Embed", has_collection)
    step("3️⃣", "Explore", has_embeddings)
    step("4️⃣", "Search", has_embeddings)


def _render_delete_section(vs: VectorStore, names: list) -> None:
    st.markdown("**Delete collection**")
    if not names:
        st.caption("No collections to delete.")
        return

    to_delete = st.selectbox("Select", options=names, key="del_select")

    confirm_key = f"confirm_delete_{to_delete}"
    confirmed = st.checkbox(f'Type to confirm: delete "{to_delete}"', key=confirm_key)

    if confirmed:
        if st.button("🗑️ Delete permanently", type="primary", key="btn_delete"):
            try:
                vs.delete_collection(to_delete)
                # Clear active selection if we just deleted it
                if st.session_state.get(ACTIVE_COLLECTION_KEY) == to_delete:
                    st.session_state[ACTIVE_COLLECTION_KEY] = None
                st.success(f"Deleted **{to_delete}**.")
                st.rerun()
            except Exception as e:
                st.error(str(e))


def _render_rename_section(vs: VectorStore, names: list) -> None:
    st.markdown("**Rename collection**")
    if not names:
        st.caption("No collections to rename.")
        return

    to_rename = st.selectbox("Select", options=names, key="ren_select")
    new_name = st.text_input("New name", key="ren_new_name").strip()

    if new_name and new_name != to_rename:
        if st.button("✏️ Rename", key="btn_rename"):
            with st.spinner("Copying data…"):
                try:
                    vs.rename_collection(to_rename, new_name)
                    if st.session_state.get(ACTIVE_COLLECTION_KEY) == to_rename:
                        st.session_state[ACTIVE_COLLECTION_KEY] = new_name
                    st.success(f"Renamed to **{new_name}**.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
