"""
Page – Samples
======================
Rebuild the 'inspiration samples' folder: one illustration per visual type
(most-recently-created wins), copied flat + nested, with an XLS index.

Demonstrates the **live log** pattern: stdlib logging records from
``sample_illustrations`` are streamed to a ``st.code`` container via
``StreamlitLogHandler``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.notion_client import load_config
from src.sample_illustrations import apply_samples, plan_samples
from src.ui_utils import attach_log_handler, detach_log_handler, open_in_explorer

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


def render() -> None:
    st.header("🗂 Samples")
    st.caption(
        "Rebuild the **inspiration samples** folder — one illustration per "
        "visual type, copied flat + nested with an XLS index."
    )

    try:
        config = load_config(str(CONFIG_PATH))
    except FileNotFoundError as e:
        st.error(f"Missing config.json: {e}")
        return

    dest_folder = Path(config.get("samples_dest_folder", ""))
    xls_output = Path(config.get("samples_xls_output", ""))
    source_folder = Path(config.get("source_folder", ""))

    with st.container(border=True):
        st.markdown("**Paths**")
        st.caption(f"Source PNGs: `{source_folder}`")
        st.caption(f"Destination: `{dest_folder}`")
        st.caption(f"XLS index: `{xls_output}`")
        if not source_folder.exists():
            st.error(f"Source folder not found: {source_folder}")
            return

    tab_plan, tab_apply = st.tabs(["Plan", "Rebuild"])

    with tab_plan:
        st.markdown("Dry-run: query Notion and preview what **would** be written. No filesystem changes.")
        if st.button("Preview plan", key="samples_plan"):
            _run_and_render(config, apply=False)

    with tab_apply:
        st.markdown(
            "⚠️ This wipes the destination folder's contents and rebuilds from scratch "
            "(the folder itself is preserved for iCloud sync)."
        )
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("Rebuild samples", type="primary", key="samples_apply"):
                _run_and_render(config, apply=True)
        with col2:
            if dest_folder.exists():
                if st.button("Open samples folder", key="samples_open"):
                    open_in_explorer(dest_folder)


def _run_and_render(config: dict, apply: bool) -> None:
    st.markdown("#### Live log")
    log_container = st.empty()
    handler = attach_log_handler(log_container)

    status = st.status(
        "Querying Notion + planning samples…",
        expanded=True,
        state="running",
    )

    try:
        plan = plan_samples(config)
        status.update(
            label=f"Planned {plan.planned} samples ({plan.missing_count} missing PNGs)",
            state="running",
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Planned samples", plan.planned)
        m2.metric("Missing PNGs", plan.missing_count)
        m3.metric("Visual types w/o illustrations", plan.skipped_no_illustrations)

        if plan.rows:
            preview_df = pd.DataFrame(plan.rows)[
                ["concept_type", "concept", "visual_type", "illustration_title", "missing_source_file"]
            ].rename(columns={"missing_source_file": "missing_png"})
            with st.expander(f"Preview planned rows ({len(preview_df)})", expanded=False):
                st.dataframe(preview_df, width="stretch", hide_index=True)

        if not apply:
            status.update(label="Preview only — nothing written.", state="complete", expanded=False)
            return

        if not plan.rows:
            status.update(label="Nothing to write.", state="error", expanded=True)
            return

        progress = st.progress(0.0, text="Copying…")

        def on_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Copying {done}/{total}…")

        summary = apply_samples(plan, config, progress_cb=on_progress)
        progress.empty()

        status.update(
            label=(
                f"Copied {summary['copied']} / {summary['planned']} "
                f"(missing: {summary['missing']})"
            ),
            state="complete",
            expanded=False,
        )
        st.success(f"XLS index written to `{summary['xls_output']}`")

    except Exception as e:
        status.update(label=f"Failed: {e}", state="error", expanded=True)
        st.exception(e)
    finally:
        detach_log_handler(handler)


render()
