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
    st.subheader("How it works")

    st.markdown(
        """
        Three offline phases feed one live search:

        1. **Metaphor enrichment** (`src/enrichment.py`) — an LLM extracts each
           illustration's underlying metaphor structure (meanings, themes, tone)
           and caches it in `index/enrichments.jsonl`.
        2. **Offline indexing** (`src/build_index.py`) — joins Notion databases,
           composes enriched embed text, embeds with `BAAI/bge-large-en-v1.5`,
           and persists the matrix + metadata to `index/`.
        3. **Interactive search** (`src/pages/search.py`) — embeds your query,
           dot-products against the stored matrix, and deduplicates results by
           visual type so the top 10 spans distinct renderings.

        See **README.md** for the full architecture, batching/caching details,
        and CLI reference.
        """
    )

    st.markdown("---")
    st.info(
        "**First time?** Open **Build** from the sidebar and run a dry-run "
        "first to verify the Notion credentials, then a full build. After that, "
        "the Search page runs offline."
    )
