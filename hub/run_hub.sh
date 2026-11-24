#!/usr/bin/env bash
# Start the IT Ops AI Suite. First run creates hub/.venv and installs the hub requirements.
set -e
cd "$(dirname "$0")/.."                      # AIProjects/
if [ ! -x hub/.venv/bin/python ]; then
  echo "Creating hub/.venv ..."
  python3 -m venv hub/.venv
  hub/.venv/bin/python -m pip install --upgrade pip -q
  hub/.venv/bin/python -m pip install -r hub/requirements.txt
fi
exec hub/.venv/bin/python -m streamlit run hub/app.py --server.port 8600 "$@"
