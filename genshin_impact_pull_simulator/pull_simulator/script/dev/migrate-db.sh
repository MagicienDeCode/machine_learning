#!/bin/bash
# Creates or updates the tables: applies the Alembic migrations (migrations/versions). Safe to
# re-run, it only applies what is missing. Usage, from anywhere (container running):
#   ./script/dev/migrate-db.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

cd "$REPO_DIR"
echo "Creating the tables (alembic upgrade head)..."
.venv/bin/python -m alembic upgrade head
