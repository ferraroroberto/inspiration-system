"""
Page – Search
=============
Paste text → rank 1,500 illustrations by cosine similarity → dedup by visual
type → show top N as cards with thumbnails and metadata.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.notion_client import load_config
from src.ui_utils import open_in_explorer

CONFIG_PATH = Path(__file__).parent.parent.parent / "config.json"


@st.cache_data(show_spinner=False)
def load_index(index_folder: str):
    folder = Path(index_folder)
    df = pd.read_parquet(folder / "illustrations.parquet")
    embeddings = np.load(folder / "embeddings.npy")
    meta = json.loads((folder / "index_meta.json").read_text(encoding="utf-8"))
    return df, embeddings, meta


@st.cache_resource(show_spinner="Loading embedding model…")
def load_model(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def index_age_days(built_at_iso: str) -> float:
    built = datetime.fromisoformat(built_at_iso)
    now = datetime.now(timezone.utc)
    return (now - built).total_seconds() / 86400.0


def rank_and_dedup(df: pd.DataFrame, embeddings: np.ndarray, query_vec: np.ndarray, k: int):
    scores = embeddings @ query_vec  # cosine since both normalized
    order = np.argsort(-scores)
    seen_vt: set = set()
    picks: list = []
    for idx in order:
        row = df.iloc[int(idx)]
        if row["missing_png"]:
            continue
        vt = row["visual_type_id"]
        if vt in seen_vt:
            continue
        seen_vt.add(vt)
        picks.append((int(idx), float(scores[int(idx)])))
        if len(picks) >= k:
            break
    return picks


def render_card(col, df_row: pd.Series, score: float) -> None:
    with col:
        with st.container(border=True):
            png_path = Path(df_row["png_path"])
            if png_path.exists():
                st.image(str(png_path), width=280)
            else:
                st.warning("PNG not found on disk")

            st.markdown(f"**{df_row['title']}**")
            breadcrumb = " / ".join(
                part for part in [df_row.get("concept_type"), df_row.get("concept_name"), df_row.get("visual_type_name")] if part
            )
            st.caption(breadcrumb)
            st.progress(min(max(score, 0.0), 1.0), text=f"match {score:.2f}")

            theme = df_row.get("theme") or ""
            if theme:
                pills = " ".join(f"`{t.strip()}`" for t in theme.split(";") if t.strip())
                st.markdown(pills)

            alt = df_row.get("alt_text") or ""
            if alt:
                with st.expander("ALT text"):
                    st.write(alt)

            btn_key = f"open_{df_row['illustration_id']}"
            if st.button("Open folder", key=btn_key):
                open_in_explorer(png_path.parent)


def render() -> None:
    st.header("🔍 Search")
    st.caption("Paste an idea, quote, or paragraph — get diverse metaphor options, one per visual type.")

    try:
        config = load_config(str(CONFIG_PATH))
    except FileNotFoundError as e:
        st.error(f"Missing config.json: {e}")
        return

    try:
        df, embeddings, meta = load_index(config["index_folder"])
    except FileNotFoundError:
        st.error(
            f"No index found at `{config['index_folder']}`. "
            "Go to **Build** to create one."
        )
        return

    age = index_age_days(meta["built_at"])
    if age > 7:
        st.warning(f"Index is {age:.0f} days old ({meta['built_at']}). Consider rebuilding.")
    else:
        st.caption(
            f"Index: {meta['count']} illustrations · model `{meta['model']}` · built {meta['built_at']}"
        )

    query_text = st.text_area(
        "Paste an idea, quote, or paragraph",
        height=200,
        placeholder="e.g. 'Rising above obstacles to reach new heights' — or a full article paragraph.",
        key="search_query_text",
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        k = st.slider("Number of suggestions", 5, 20, 10, key="search_topk")
    with c2:
        go = st.button("Find illustrations", type="primary", key="search_go")

    if not go:
        return
    if not query_text.strip():
        st.warning("Paste some text first.")
        return

    model = load_model(meta["model"])
    with st.spinner("Embedding query…"):
        # bge-large-en-v1.5 recommends prefixing retrieval queries with this
        # instruction. It is applied only on the query side; docs are embedded
        # without a prefix at index time. For non-bge models the prefix adds a
        # few tokens but doesn't hurt retrieval quality.
        query_for_embed = (
            "Represent this sentence for searching relevant passages: " + query_text
            if "bge" in meta["model"].lower()
            else query_text
        )
        q_vec = model.encode([query_for_embed], normalize_embeddings=True)[0].astype(np.float32)

    picks = rank_and_dedup(df, embeddings, q_vec, k)
    if not picks:
        st.info("No matches found.")
        return

    st.markdown(f"### Top {len(picks)} — one per visual type")

    cols_per_row = 2
    for i in range(0, len(picks), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (idx, score) in enumerate(picks[i : i + cols_per_row]):
            render_card(cols[j], df.iloc[idx], score)


render()
