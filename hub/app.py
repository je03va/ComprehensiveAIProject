"""
IT Ops AI Suite - integrated menu GUI for the semester's AI products.

Run from the AIProjects folder:
    ./hub/run_hub.sh            (first run creates hub/.venv)
or  streamlit run hub/app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from registry import MODULES, SECTIONS, SUITE_NAME  # noqa: E402

st.set_page_config(page_title=SUITE_NAME, page_icon=":material/hub:", layout="wide")

PAGES_DIR = Path(__file__).resolve().parent / "views"

home = st.Page(str(PAGES_DIR / "home.py"), title="Home", icon=":material/home:", default=True)
nav = {"Suite": [home]}
for section in SECTIONS:
    nav[section] = [st.Page(str(PAGES_DIR / f"{m['key']}.py"), title=m["title"], icon=m["icon"],
                            url_path=m["key"])
                    for m in MODULES if m["section"] == section]

with st.sidebar:
    st.markdown(f"### {SUITE_NAME}")
    st.caption("CSC 470 · AI Integrative Project · Jalen Elliott")

st.navigation(nav, expanded=True).run()
