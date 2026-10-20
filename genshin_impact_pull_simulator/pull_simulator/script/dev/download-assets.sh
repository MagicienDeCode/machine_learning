#!/bin/bash
# Downloads the character and weapon pictures into static/ (local use only, git-ignored), then
# re-seeds the catalogue so the API serves them. Pictures already there are skipped.
# Usage, from anywhere:
#   ./script/dev/download-assets.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"
.venv/bin/python -m app.tools.download_assets
