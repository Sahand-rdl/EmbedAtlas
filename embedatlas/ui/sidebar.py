"""
EmbedAtlas — Sidebar
"""

from __future__ import annotations

import streamlit as st

from embedatlas.config import APP_ICON, APP_TITLE, APP_VERSION
from embedatlas.core.vectorstore import VectorStore

ACTIVE_COLLECTION_KEY = "active_collection"
VS_KEY = "vectorstore"


def _get_vs() -> VectorStore:
    if VS_KEY not in st.session_state:
        st.session_state[VS_KEY] = VectorStore()
    return st.session_state[VS_KEY]


def render_sidebar() -> None:
    vs = _get_vs()

    with st.sidebar:
        # ── Brand ─────────────────────────────────────────────────────
        st.markdown(f"# {APP_ICON} {APP_TITLE}")
        st.caption(f"v{APP_VERSION}")
        st.divider()

        # ── Collection selector ───────────────────────────────────────
        st.markdown("### Collections")

        collections = vs.list_collections()
        names = [c.name for c in collections]

        if not names:
            st.info("No collections yet.\nHead to **Ingest** to create one.")
            st.session_state[ACTIVE_COLLECTION_KEY] = None
        else:
            prev = st.session_state.get(ACTIVE_COLLECTION_KEY)
            default_idx = names.index(prev) if prev in names else 0

            selected = st.selectbox(
                "Active collection",
                options=names,
                index=default_idx,
                label_visibility="collapsed",
            )
            st.session_state[ACTIVE_COLLECTION_KEY] = selected

            info = next(c for c in collections if c.name == selected)
            _render_collection_card(info, selected)

        st.divider()

        # ── Collection management ─────────────────────────────────────
        with st.expander("⚙️  Manage collections", expanded=False):
            _render_delete_section(vs, names)
            st.divider()
            _render_rename_section(vs, names)

        # ── Footer ────────────────────────────────────────────────────
        st.markdown(
            "<br><br><div style='font-size:11px; color:gray;'>"
            "EmbedAtlas · open source<br>"
            "<a href='https://github.com/your-org/embedatlas'>GitHub</a>"
            "</div>",
            unsafe_allow_html=True,
        )


def _render_collection_card(info, collection_name: str) -> None:
    # Check session state for model_id (set by embed page after successful run)
    # Fall back to ChromaDB collection metadata
    model_id = st.session_state.get(f"model_id_{collection_name}") or info.metadata.get(
        "model_id"
    )

    if info.count > 0 and model_id:
        model_short = model_id.split("/")[-1]
        status_color = "#4ade80"  # green
        status_text = f"{info.count:,} chunks · {model_short}"
        status_icon = "✅"
    elif info.count > 0:
        status_color = "#4ade80"
        status_text = f"{info.count:,} chunks · embedded"
        status_icon = "✅"
    else:
        status_color = "gray"
        status_text = "0 chunks · not embedded yet"
        status_icon = "⏳"

    st.markdown(
        f"""
        <div style="
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 8px;
            padding: 10px 14px;
            margin-top: 6px;
        ">
            <span style="font-size:13px; font-weight:600;">{info.name}</span>
            <span style="float:right">{status_icon}</span><br>
            <span style="font-size:11px; color:{status_color};">
                {status_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if info.metadata_keys:
        st.caption("Fields: " + "  `" + "`  `".join(info.metadata_keys) + "`")


def _render_delete_section(vs: VectorStore, names: list) -> None:
    st.markdown("**Delete collection**")
    if not names:
        st.caption("No collections to delete.")
        return

    to_delete = st.selectbox("Select", options=names, key="del_select")
    confirm_key = f"confirm_delete_{to_delete}"
    confirmed = st.checkbox(f'Confirm: delete "{to_delete}"', key=confirm_key)

    if confirmed:
        if st.button("🗑️ Delete permanently", type="primary", key="btn_delete"):
            try:
                vs.delete_collection(to_delete)
                if st.session_state.get(ACTIVE_COLLECTION_KEY) == to_delete:
                    st.session_state[ACTIVE_COLLECTION_KEY] = None
                # Clear cached model_id for deleted collection
                st.session_state.pop(f"model_id_{to_delete}", None)
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
                    # Migrate cached model_id key
                    old_key = f"model_id_{to_rename}"
                    if old_key in st.session_state:
                        st.session_state[f"model_id_{new_name}"] = st.session_state.pop(
                            old_key
                        )
                    st.success(f"Renamed to **{new_name}**.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
