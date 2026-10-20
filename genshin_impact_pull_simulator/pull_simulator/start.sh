#!/bin/bash
# Starts the backend, from scratch if needed: creates the venv, starts MySQL, brings the database
# up to date (tables, catalogue, test login), fetches the pictures once, then runs the API.
#
# Usage, from anywhere (Docker must be running):
#   ./start.sh                 # http://localhost:8000 (override with PORT=...), auto-reload on
#   ./start.sh --reset         # drops the database first: every player starts over
#   ./start.sh --no-assets     # skips the picture download (the front then shows placeholders)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"
PORT="${PORT:-8000}"

RESET_ARG=()
ASSETS=true
for arg in "$@"; do
  case "$arg" in
    --reset) RESET_ARG=(--reset) ;;
    --no-assets) ASSETS=false ;;
    *) echo "Usage: $0 [--reset] [--no-assets]" >&2; exit 1 ;;
  esac
done

if [ ! -x .venv/bin/python ]; then
  PYTHON="$(command -v python3.13 || command -v python3.12 || command -v python3)"
  echo "Creating the virtual env with $PYTHON..."
  "$PYTHON" -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -e . pytest ruff
fi

./script/dev/init-db.sh "${RESET_ARG[@]+"${RESET_ARG[@]}"}"

# Only when pictures are missing (the currency icons came last): the download skips what is
# there, but still asks Project Amber for its lists.
if [ "$ASSETS" = true ] && [ ! -f static/currency/stardust.png ]; then
  ./script/dev/download-assets.sh
fi

echo
echo "API on http://localhost:$PORT (docs: /docs). Login traveler@test.local / 123456."
exec .venv/bin/uvicorn app.main:app --reload --port "$PORT"
