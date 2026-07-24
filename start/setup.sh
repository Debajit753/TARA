#!/usr/bin/env bash
# Rebuild the TARA environment on any Mac/Linux laptop. Needs internet once.
# Usage:  ./setup.sh
set -e
cd "$(dirname "$0")"
echo "Creating virtual environment (.venv) ..."
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
echo "Installing dependencies from requirements.txt (this can take a few minutes) ..."
.venv/bin/pip install -r requirements.txt
echo ""
echo "Done. Test it with:   .venv/bin/python quickstart.py"
echo "Activate manually:    source .venv/bin/activate"
