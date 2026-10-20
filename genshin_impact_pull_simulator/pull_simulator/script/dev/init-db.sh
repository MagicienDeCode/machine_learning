#!/bin/bash
# One command to get a ready-to-use local database: starts the MySQL container, creates the
# tables, loads the catalogue and seeds the test login. Chains the smaller scripts of this folder.
#
# Usage, from anywhere (Docker must be running, the venv must exist -- start.sh creates it):
#   ./script/dev/init-db.sh           # keeps whatever data is already there
#   ./script/dev/init-db.sh --reset   # drops the database first: every table starts empty
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

CONTAINER_NAME="genshin-wish-mysql"
DB_NAME="genshinwish"
DB_USER="genshinwish"
DB_PASSWORD="genshinwish"

RESET=false
if [ "${1:-}" = "--reset" ]; then
  RESET=true
elif [ $# -gt 0 ]; then
  echo "Usage: $0 [--reset]" >&2
  exit 1
fi

"$SCRIPT_DIR/docker-db.sh"
if [ "$RESET" = true ]; then
  "$SCRIPT_DIR/create-db.sh"
fi
"$SCRIPT_DIR/migrate-db.sh"
"$SCRIPT_DIR/seed-catalog.sh"
"$SCRIPT_DIR/seed-test-users.sh"

echo
echo "Tables in '$DB_NAME':"
docker exec "$CONTAINER_NAME" mysql -u"$DB_USER" -p"$DB_PASSWORD" "$DB_NAME" -N -e "SHOW TABLES;" \
  2>/dev/null | sed 's/^/  /'
