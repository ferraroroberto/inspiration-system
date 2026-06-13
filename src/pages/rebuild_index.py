"""
Page – Build
==================
Pull the latest Notion data, embed every illustration, and persist the
snapshot to ``index/``. A full build embeds ~1,500 rows — long enough that
watching a live log matters.

Uses ``StreamlitLogHandler`` to pipe stdlib logging from
``build_index.build_index`` into a live ``st.code`` container.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from src.build_index import build_index
from src.notion_client import load_config
from src.ui_utils import attach_log_handler, detach_log_handler, open_in_explorer

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


def _read_existing_meta(index_folder: Path) -> dict | None:
    meta_path = index_folder / "index_meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def render() -> None:
    st.header("🧠 Build")
    st.caption(
        "Query Notion, compose per-illustration text, embed with "
        "sentence-transformers, and persist the snapshot to `index/`."
    )

    try:
        config = load_config(str(CONFIG_PATH))
    except FileNotFoundError as e:
        st.error(f"Missing config.json: {e}")
        return

    index_folder = Path(config["index_folder"])
    model_name = config.get("embed_model", "sentence-transformers/all-MiniLM-L6-v2")

    with st.container(border=True):
        st.markdown("**Settings**")
        st.caption(f"Model: `{model_name}`")
        st.caption(f"Index folder: `{index_folder}`")

        existing = _read_existing_meta(index_folder)
        if existing:
            built = existing.get("built_at", "?")
            try:
                age = (
                    datetime.now(timezone.utc) - datetime.fromisoformat(built)
                ).total_seconds() / 86400.0
                age_txt = f"{age:.1f} days old"
            except Exception:
                age_txt = "age unknown"
            st.caption(
                f"Current snapshot: {existing.get('count', '?')} rows · "
                f"built {built} ({age_txt})"
            )
        else:
            st.caption("No index built yet.")

    tab_dry, tab_full = st.tabs(["Dry run", "Full build"])

    with tab_dry:
        st.markdown(
            "Query Notion and report counts — **no embedding, no files written**. "
            "Good for verifying the Notion credentials and database IDs."
        )
        if st.button("Run dry-run", key="build_dry"):
            _run(config, dry_run=True)

    with tab_full:
        st.markdown(
            "Full build: query Notion, run metaphor enrichment (cached — "
            "only new rows incur hub/LLM calls), embed ~1,500 rows, "
            "write `illustrations.parquet`, `embeddings.npy`, and "
            "`index_meta.json`. Takes **30–90 seconds** on CPU after "
            "enrichment (first run also downloads the model)."
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Build", type="primary", key="build_full"):
                _run(config, dry_run=False)
        with col2:
            if index_folder.exists():
                if st.button("Open index folder", key="build_open"):
                    open_in_explorer(index_folder)


def _run(config: dict, dry_run: bool) -> None:
    st.markdown("#### Live log")
    log_container = st.empty()
    handler = attach_log_handler(log_container)

    status = st.status(
        "Dry run — querying Notion…" if dry_run else "Querying Notion + embedding…",
        expanded=True,
        state="running",
    )
    progress = st.progress(0.0, text="Waiting…")

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total, text=f"Embedding {done}/{total}…")

    try:
        summary = build_index(config, dry_run=dry_run, progress_cb=on_progress)
        progress.empty()
        if dry_run:
            status.update(
                label=f"Dry run complete — {summary['planned']} rows would be embedded.",
                state="complete",
                expanded=False,
            )
        else:
            # Invalidate cached load_index in the Search page
            st.cache_data.clear()
            status.update(
                label=(
                    f"Build complete — {summary['count']} rows in "
                    f"{summary['elapsed_s']:.1f}s (missing PNG: {summary['missing_png']})"
                ),
                state="complete",
                expanded=False,
            )
            st.success(f"Parquet: `{summary['parquet_path']}`")
            st.success(f"Embeddings: `{summary['npy_path']}`")
            st.success(f"Meta: `{summary['meta_path']}`")

    except Exception as e:
        progress.empty()
        status.update(label=f"Failed: {e}", state="error", expanded=True)
        st.exception(e)
    finally:
        detach_log_handler(handler)


render()
