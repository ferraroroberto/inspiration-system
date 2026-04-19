"""
Inspiration Illustration Finder — Main Entry
============================================
Launch with:
    streamlit run app/app.py

Pages live under ``src/pages/`` and are registered explicitly via
``st.navigation`` + ``st.Page`` — this lets us keep all page scripts
inside ``src/`` while the launcher lives under ``app/``.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import streamlit as st

from src.menu import PAGES, render_home

st.set_page_config(
    page_title="Illustration Inspiration Finder",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Default (dark) theme lives in ``.streamlit/config.toml`` — that's Streamlit's
# native mechanism, read once at startup. Light mode is a runtime CSS overlay
# in ``app/styles/light.css``, injected when the sidebar toggle is on.
_STYLES_DIR = Path(__file__).resolve().parent / "styles"


def _inject_css(filename: str) -> None:
    css = (_STYLES_DIR / filename).read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


with st.sidebar:
    if st.toggle("☀  Light mode", value=False, key="light_mode"):
        _inject_css("light.css")

_PAGES_DIR = Path(__file__).parent.parent / "src" / "pages"

nav_pages = [
    st.Page(render_home, title="Home", icon="🏠", default=True),
] + [
    st.Page(str(_PAGES_DIR / p["file"]), title=p["label"], icon=p["icon"])
    for p in PAGES
]

pg = st.navigation(nav_pages, position="sidebar")
pg.run()
