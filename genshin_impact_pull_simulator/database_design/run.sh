#!/bin/bash
# Runs the lesson's SQL files against the local MySQL container, into the genshinwish_lesson
# database (the app's own genshinwish database is never touched).
#
# Usage, from anywhere (the container must be running: pull_simulator/script/dev/docker-db.sh):
#   ./run.sh                       # steps 00 to 11, one after the other
#   ./run.sh 02_catalogue.sql      # one step only (or several, in the order given)
#   ./run.sh 99_drop_all.sql       # drop every table
#   ./run.sh --shell               # a mysql prompt inside genshinwish_lesson
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="genshin-wish-mysql"
ROOT_PASSWORD="root"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Error: the mysql container isn't running. Start it first:" >&2
  echo "  ../pull_simulator/script/dev/docker-db.sh" >&2
  exit 1
fi

if [ "${1:-}" = "--shell" ]; then
  exec docker exec -it "$CONTAINER_NAME" mysql -uroot -p"$ROOT_PASSWORD" --default-character-set=utf8mb4 genshinwish_lesson
fi

if [ $# -gt 0 ]; then
  FILES=("$@")
else
  FILES=()
  for f in "$DIR"/[0-9][0-9]_*.sql; do
    case "$(basename "$f")" in 99_*) ;; *) FILES+=("$(basename "$f")") ;; esac
  done
fi

for file in "${FILES[@]}"; do
  echo "=== $file"
  # --table prints SELECT results as tables; -vvv would also echo each statement.
  docker exec -i "$CONTAINER_NAME" mysql -uroot -p"$ROOT_PASSWORD" --default-character-set=utf8mb4 \
    --table < "$DIR/$(basename "$file")" 2>&1 | { grep -v "Using a password on the command line" || true; }
done
