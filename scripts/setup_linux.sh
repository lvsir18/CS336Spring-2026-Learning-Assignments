#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# RTX PRO 6000 Blackwell requires a compatible architecture build and driver.
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e '.[alignment,data,plots]'
python -m pip freeze > requirements-server-lock.txt
python scripts/server_check.py
