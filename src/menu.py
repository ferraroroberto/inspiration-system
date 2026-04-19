"""
Menu / Page Registry
====================
Central registry of every page in this app. Each entry maps a label to the
module under ``pages/`` that implements it. The same registry drives the
home page's card grid.

Adding a new page:
  1. Create ``pages/my_page.py`` with a top-level script body (Streamlit
     runs each file in ``pages/`` as its own script).
  2. Append an entry to ``PAGES``.
"""

from __future__ import annotations

import streamlit as st

PAGES: list[dict] = [
    {
        "label": "Search",
        "icon": "🔍",
        "description": "Paste an idea, quote, or paragraph — get 10 ranked illustration suggestions, one per visual type.",
        "file": "search.py",
    },
    {
        "label": "Samples",
        "icon": "🗂",
        "description": "Rebuild the 'inspiration samples' folder: one PNG per visual type, flat + nested, with an XLS index.",
        "file": "curated_samples.py",
    },
    {
        "label": "Build",
        "icon": "🧠",
        "description": "Pull the latest Notion data, run metaphor enrichment (or just the enrichment pass on its own), and re-embed everything. Live log + progress bar.",
        "file": "rebuild_index.py",
    },
]


def render_home() -> None:
    """Render the landing / home page with an overview of all tools."""
    st.header("🎨 Illustration Inspiration Finder")
    st.markdown(
        """
        Paste text → get **diverse metaphor options** from your Notion-archived
        illustrations, one per visual type.

        Use the **sidebar** to navigate between the tools below. The Notion
        data is pulled into a local embedding index once, then searches run
        fully offline in milliseconds.
        """
    )

    st.markdown("---")
    st.subheader("Available tools")

    cols = st.columns(len(PAGES))
    for col, page in zip(cols, PAGES):
        with col:
            st.markdown(f"#### {page['icon']} {page['label']}")
            st.write(page["description"])

    st.markdown("---")
    st.subheader("Tech stack & how it works")

    st.markdown(
        """
        Three phases. The first two run offline and incrementally; the third
        runs live per query.

        **1 · Metaphor enrichment** — `src/enrichment.py`
        For each illustration, **Claude Code in headless mode**
        (`claude -p --output-format json`) returns a structured JSON object with
        *visual elements*, *metaphorical meanings*, *applicable themes*, *tone*,
        and *abstraction level*. The prompt asks for the underlying metaphor
        structure, not the literal scene — so the embedding later keys off
        *what an illustration could represent*, not just what it shows.
        Batched 15 rows per call, cached in `index/enrichments.jsonl`,
        resumable and idempotent. Uses the existing Claude Code auth — no
        separate API key, no per-call billing.

        **2 · Offline indexing** — `src/build_index.py`
        Pulls the three **Notion** databases (concepts / visual types /
        illustrations) via the Notion API, joins them, runs enrichment for
        any uncached rows, composes the embed text with enriched fields
        *first* (early tokens carry more weight), and embeds with
        **`BAAI/bge-large-en-v1.5`** (sentence-transformers, 1024-dim unit
        vectors). Persists three files to `index/`:
        `illustrations.parquet` (metadata + enriched columns),
        `embeddings.npy` (float32 matrix, ~6 MB for 1.5k rows), and
        `index_meta.json` (model, dim, count, built_at).

        **3 · Interactive search** — `src/pages/search.py`
        Prefixes the query with bge's retrieval instruction (query-side only),
        embeds it (sub-100ms on CPU), dot-products against the stored matrix
        for cosine similarity (sub-ms over 1.5k rows), then walks the ranked
        list keeping only the highest-scoring illustration per *visual type*
        so the top 10 is diverse by rendering, not ten near-identical mountains.

        **Samples** — `src/sample_illustrations.py`
        A side utility that materialises a browsable "inspiration samples"
        folder on disk: one PNG per *visual type* (most recently created
        wins), laid out both **flat** (all PNGs in one folder) and **nested**
        (grouped by concept / visual type), plus an XLS index
        (`samples_index.xlsx`) listing every selected row. Runs in two
        modes — **Plan** (dry-run, logs what would be written without
        touching disk) and **Rebuild** (wipes the destination and writes
        the new layout). Shares Notion access + field extraction with
        `build_index.py` via `src/notion_client.py`; only the
        sample-selection + filesystem-layout logic is specific to this
        tool. Missing PNGs are logged but don't abort — rows are still
        written to the XLS index for debugging.

        **UI layer** — **Streamlit** with explicit `st.navigation` + `st.Page`
        registration (pages live under `src/pages/`, registered in
        `src/menu.py`). Long-running jobs stream into the UI via a custom
        `StreamlitLogHandler` (stdlib `logging.Handler` that rewrites an
        `st.empty()` container on each record). Dark theme in
        `.streamlit/config.toml`; light mode is a runtime CSS overlay from
        `app/styles/light.css`, toggled in the sidebar.

        **Two diversity mechanisms, one at each end.** Enrichment opens up
        *metaphor* diversity in the embedding space; the visual-type dedup in
        the search walk enforces *rendering* diversity in the final top-N.
        Once built, the index is fully self-contained — Notion and `claude -p`
        aren't touched again until you rebuild.
        """
    )

    st.markdown("---")
    st.info(
        "**First time?** Open **Build** from the sidebar and run a dry-run "
        "first to verify the Notion credentials, then a full build. After that, "
        "the Search page runs offline."
    )
