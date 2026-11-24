"""
Process manager for modules that run outside the hub.

- "streamlit" modules keep their own virtual environment (the NLP sentiment module
  needs NumPy 1.x, the regression modules NumPy 2.x), so each one is started as its
  own Streamlit server on a fixed port and embedded in the hub with an iframe.
- "desktop" modules (turtle games) open their own window.

Every child process runs with its working directory set to the module folder, so
the module's relative paths (data/, outputs/, logs/) work exactly as they do when
the module is run on its own.
"""
import atexit
import glob
import os
import socket
import subprocess
import sys
import time

import streamlit as st

from registry import ROOT, module_dir


def _venv_python(folder):
    py = folder / ".venv" / "bin" / "python"
    if sys.platform == "win32":
        py = folder / ".venv" / "Scripts" / "python.exe"
    return py if py.exists() else None


def _has_package(python, package):
    venv = python.parent.parent
    patterns = [str(venv / "lib" / "python*" / "site-packages" / package),
                str(venv / "Lib" / "site-packages" / package)]
    return any(glob.glob(p) for p in patterns)


def pick_python(m, needs=None):
    """The module's own .venv if it can run the module, otherwise the hub's Python."""
    own = _venv_python(module_dir(m))
    if own and (needs is None or _has_package(own, needs)):
        return str(own), "module .venv"
    return sys.executable, "hub Python"


def port_open(port, host="127.0.0.1"):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.3)
        return s.connect_ex((host, port)) == 0


@st.cache_resource
def _processes():
    """One shared table of child processes for the whole hub server."""
    procs = {}

    def _cleanup():
        for p in procs.values():
            if p["proc"].poll() is None:
                p["proc"].terminate()
    atexit.register(_cleanup)
    return procs


def status(m):
    p = _processes().get(m["key"])
    if p and p["proc"].poll() is None:
        return "running"
    if m["kind"] == "streamlit" and port_open(m["port"]):
        return "running (external)"
    if p:
        return f"stopped (exit code {p['proc'].returncode})"
    return "not started"


def is_running(m):
    return status(m).startswith("running")


def log_path(m):
    logs = ROOT / "hub" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    return logs / f"{m['key']}.log"


def start(m):
    if is_running(m):
        return
    folder = module_dir(m)
    if m["kind"] == "streamlit":
        python, source = pick_python(m, needs="streamlit")
        cmd = [python, "-m", "streamlit", "run", m["entry"],
               "--server.port", str(m["port"]), "--server.headless", "true",
               "--browser.gatherUsageStats", "false", "--server.runOnSave", "false"]
    else:
        python, source = pick_python(m)
        cmd = [python, m["entry"]]
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    env.pop("VIRTUAL_ENV", None)
    log = open(log_path(m), "w")
    proc = subprocess.Popen(cmd, cwd=folder, stdout=log, stderr=subprocess.STDOUT, env=env)
    _processes()[m["key"]] = {"proc": proc, "python": python, "source": source, "started": time.time()}


def stop(m):
    p = _processes().get(m["key"])
    if p and p["proc"].poll() is None:
        p["proc"].terminate()
        try:
            p["proc"].wait(timeout=5)
        except subprocess.TimeoutExpired:
            p["proc"].kill()


def wait_until_up(m, timeout=45):
    """Wait for an embedded Streamlit server to accept connections."""
    end = time.time() + timeout
    while time.time() < end:
        if port_open(m["port"]):
            return True
        p = _processes().get(m["key"])
        if p and p["proc"].poll() is not None:
            return False
        time.sleep(0.5)
    return False


def info(m):
    return _processes().get(m["key"])


def tail_log(m, lines=40):
    path = log_path(m)
    if not path.exists():
        return ""
    return "".join(path.read_text(errors="replace").splitlines(True)[-lines:])
