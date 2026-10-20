#!/bin/bash
# Runs seed-test-users.sql against the local dev MySQL container (see that file's header for what
# it does and why it's safe to re-run). Usage, from anywhere:
#   ./script/dev/seed-test-users.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="$SCRIPT_DIR/seed-test-users.sql"

CONTAINER_NAME="genshin-wish-mysql"
DB_USER="genshinwish"
DB_PASSWORD="genshinwish"
DB_NAME="genshinwish"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Error: the mysql container isn't running. Start it first:" >&2
  echo "  ./script/dev/docker-db.sh" >&2
  exit 1
fi

echo "Seeding test users into '$DB_NAME' (container $CONTAINER_NAME)..."
docker exec -i "$CONTAINER_NAME" mysql -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" < "$SQL_FILE" 2>/dev/null

echo "Done. Login (password 123456):"
docker exec "$CONTAINER_NAME" mysql -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" \
  -e "SELECT email, status FROM users WHERE email = 'traveler@test.local';" 2>/dev/null
