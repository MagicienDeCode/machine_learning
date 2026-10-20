#!/bin/bash
# Drops and recreates the empty local dev database. Usage, from anywhere (the container must be
# running, see docker-db.sh):
#   ./script/dev/create-db.sh
set -euo pipefail

CONTAINER_NAME="genshin-wish-mysql"
DB_NAME="genshinwish"
DB_USER="genshinwish"
ROOT_PASSWORD="root"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Error: the mysql container isn't running. Start it first:" >&2
  echo "  ./script/dev/docker-db.sh" >&2
  exit 1
fi

echo "Dropping and recreating database '$DB_NAME'..."
docker exec -i "$CONTAINER_NAME" mysql -uroot -p"$ROOT_PASSWORD" -e \
  "DROP DATABASE IF EXISTS \`$DB_NAME\`; CREATE DATABASE \`$DB_NAME\`;
   GRANT ALL PRIVILEGES ON \`$DB_NAME\`.* TO '$DB_USER'@'%'; FLUSH PRIVILEGES;" 2>/dev/null
echo "Done. '$DB_NAME' is empty."
