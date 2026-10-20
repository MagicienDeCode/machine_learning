#!/bin/bash
# Loads the catalogue (data/items.json, data/banners.json) into items, banners and banner_items.
# Safe to re-run: rows are upserted, so wishes and inventory keep pointing at the same items.
# Usage, from anywhere (tables created, see migrate-db.sh):
#   ./script/dev/seed-catalog.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"
echo "Loading the catalogue..."
.venv/bin/python -m app.tools.seed_catalog
