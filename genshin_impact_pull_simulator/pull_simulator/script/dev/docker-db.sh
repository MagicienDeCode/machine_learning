#!/bin/bash
# Starts (or creates) the local MySQL container with plain `docker` commands -- no docker-compose --
# and waits for it to report healthy. Usage, from anywhere:
#   ./script/dev/docker-db.sh
set -euo pipefail

CONTAINER_NAME="genshin-wish-mysql"
VOLUME_NAME="genshin_wish_mysql_data"
IMAGE="mysql:8"
DB_NAME="genshinwish"
DB_USER="genshinwish"
DB_PASSWORD="genshinwish"
ROOT_PASSWORD="root"
# Not 3306 nor 3307 (anti-gamble), so all of them can run side by side.
HOST_PORT="3308"

if docker ps -a --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Starting existing '$CONTAINER_NAME' container..."
  docker start "$CONTAINER_NAME" >/dev/null
else
  echo "Creating '$CONTAINER_NAME' container..."
  docker volume create "$VOLUME_NAME" >/dev/null
  docker run -d \
    --name "$CONTAINER_NAME" \
    --restart unless-stopped \
    -e MYSQL_DATABASE="$DB_NAME" \
    -e MYSQL_USER="$DB_USER" \
    -e MYSQL_PASSWORD="$DB_PASSWORD" \
    -e MYSQL_ROOT_PASSWORD="$ROOT_PASSWORD" \
    -p "$HOST_PORT:3306" \
    -v "$VOLUME_NAME:/var/lib/mysql" \
    "$IMAGE" --character-set-server=utf8mb4 --collation-server=utf8mb4_0900_ai_ci >/dev/null
fi

echo "Waiting for mysql to report healthy..."
for _ in $(seq 1 30); do
  # mysqladmin ping answers before the init scripts have created the user: check it can log in.
  if docker exec "$CONTAINER_NAME" mysql -u"$DB_USER" -p"$DB_PASSWORD" -e "SELECT 1" >/dev/null 2>&1; then
    echo "mysql is healthy."
    exit 0
  fi
  sleep 2
done

echo "Error: mysql did not become healthy in time. Check 'docker logs $CONTAINER_NAME'." >&2
exit 1
