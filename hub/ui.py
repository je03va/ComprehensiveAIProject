"""Shared UI pieces: the title/description pane, module loading, embedded and desktop pages."""
import importlib.util
import sys
import types

import streamlit as st
import streamlit.components.v1 as components

import launcher
from registry import THEME, module_dir

KIND_LABEL = {"native": "Runs inside the hub", "streamlit": "Embedded app (own environment)",
              "desktop": "Desktop window"}


def header(m):
    """Title / description pane shown at the top of every module."""
    st.title(m["title"])
    st.markdown(f":blue-badge[{m['technique']}] :green-badge[Theme: {THEME}] "
                f":gray-badge[{KIND_LABEL[m['kind']]}]")
    with st.container(border=True):
        left, right = st.columns([3, 2], gap="large")
        with left:
            st.markdown("**What it does**")
            st.write(m["description"])
            st.markdown("**Improvements for the integrated suite**")
            st.markdown("\n".join(f"- {e}" for e in m["enhancements"]))
        with right:
            st.markdown("**PEAS**")
            st.markdown("\n".join(f"- **{k}:** {v}" for k, v in m["peas"].items()))
            st.caption(f"Source: `{m['folder']}/{m['entry']}`")


def load_module(m, unique_name, strip_from=None):
    """Import a module's file by path under a unique name (several modules are called agent.py).

    strip_from: for scripts that run their simulation at import time, the line where the
    driver code starts; everything from there on is left out so only the classes load.
    """
    path = module_dir(m) / m["entry"]
    if unique_name in sys.modules:
        return sys.modules[unique_name]
    if strip_from is None:
        spec = importlib.util.spec_from_file_location(unique_name, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[unique_name] = mod
        spec.loader.exec_module(mod)
        return mod
    source = path.read_text()
    cut = source.find(strip_from)
    if cut != -1:
        source = source[:cut]
    mod = types.ModuleType(unique_name)
    mod.__file__ = str(path)
    sys.modules[unique_name] = mod
    exec(compile(source, str(path), "exec"), mod.__dict__)
    return mod


def embedded_page(m):
    header(m)
    if launcher.status(m) == "not started":
        launcher.start(m)          # start automatically the first time the page is opened
    url = f"http://localhost:{m['port']}"
    c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
    if c1.button("Restart" if launcher.is_running(m) else "Start", key=f"start_{m['key']}"):
        launcher.stop(m)
        launcher.start(m)
        st.rerun()
    if c2.button("Stop", key=f"stop_{m['key']}", disabled=not launcher.is_running(m)):
        launcher.stop(m)
        st.rerun()
    c3.link_button("Open in new tab", url)
    proc = launcher.info(m)
    c4.caption(f"Status: **{launcher.status(m)}** · port {m['port']}"
               + (f" · {proc['source']}: `{proc['python']}`" if proc else ""))

    if not launcher.status(m).startswith("running"):
        st.info("Module is stopped. Press Start to run it.")
        return
    with st.spinner(f"Starting {m['title']}..."):
        up = launcher.wait_until_up(m)
    if not up:
        st.error("The module did not start. Last lines of its log:")
        st.code(launcher.tail_log(m) or "(no output)")
        return
    st.caption("Heavy modules train or load their model the first time they open; give them a moment.")
    if hasattr(st, "iframe"):
        st.iframe(url, height=1100)
    else:
        components.iframe(url, height=1100, scrolling=True)
    with st.expander("Module log"):
        st.code(launcher.tail_log(m) or "(no output yet)")


def desktop_page(m, how_to_play):
    header(m)
    st.subheader("Play")
    st.write(how_to_play)
    running = launcher.is_running(m)
    c1, c2, c3 = st.columns([1, 1, 3])
    if c1.button("Launch game window", type="primary", disabled=running, key=f"go_{m['key']}"):
        launcher.start(m)
        st.rerun()
    if c2.button("Close game", disabled=not running, key=f"close_{m['key']}"):
        launcher.stop(m)
        st.rerun()
    c3.caption(f"Status: **{launcher.status(m)}**")
    if running:
        st.info("The game opened in its own window (check your dock / taskbar if it is behind this browser).")
    if launcher.status(m).startswith("stopped"):
        with st.expander("Game output"):
            st.code(launcher.tail_log(m) or "(no output)")
