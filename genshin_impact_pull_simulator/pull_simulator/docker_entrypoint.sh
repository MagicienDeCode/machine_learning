#!/bin/sh
# Brings the database up to date (tables + catalogue), fetches the pictures once, then runs the API.
#   SKIP_ASSETS=1   skips the picture download (the front then shows placeholders)
set -e

echo "Waiting for MySQL at ${DB_HOST:-localhost}:${DB_PORT:-3308}..."
i=0
until python -c "from app.db import engine; engine.connect().close()" 2>/dev/null; do
    i=$((i + 1))
    if [ "$i" -ge 30 ]; then
        echo "Error: cannot reach MySQL." >&2
        exit 1
    fi
    sleep 2
done

echo "Creating the tables (alembic upgrade head)..."
python -m alembic upgrade head

# static/ is a Docker volume: the download only happens the first time.
if [ "${SKIP_ASSETS:-0}" != "1" ] && [ ! -f static/currency/stardust.png ]; then
    echo "Downloading the pictures (first run only)..."
    python -m app.tools.download_assets || echo "Warning: picture download failed, continuing without them."
fi

echo "Loading the catalogue..."
python -m app.tools.seed_catalog

echo "Starting the API on :8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
