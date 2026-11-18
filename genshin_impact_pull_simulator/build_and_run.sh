#!/bin/sh
# Builds the backend (pull_simulator) and front (pull_simulator_front) images and runs them.
# MySQL must already be running in the 'genshin-wish-mysql' container (host port 3308,
# see pull_simulator/script/dev/docker-db.sh).
#   ./build_and_run.sh          build + run
#   ./build_and_run.sh build    only build
#   ./build_and_run.sh run      only run (images already built)
#   ./build_and_run.sh stop     stop and remove the two containers
# Env: FRONT_PORT (default 8080), BACKEND_PORT (default 8000), SKIP_ASSETS=1 (no picture download)
set -e
cd "$(dirname "$0")"

NETWORK=genshin-wish-net
DB_CONTAINER=genshin-wish-mysql
BACKEND_IMAGE=genshin-wish-backend
BACKEND_CONTAINER=genshin-wish-backend   # nginx.conf proxies to this name
FRONT_IMAGE=genshin-wish-front
FRONT_CONTAINER=genshin-wish-front
STATIC_VOLUME=genshin_wish_static        # downloaded pictures, kept across runs
FRONT_PORT="${FRONT_PORT:-8080}"
BACKEND_PORT="${BACKEND_PORT:-8000}"

build() {
    echo "Building backend image..."
    docker build -t "$BACKEND_IMAGE" pull_simulator
    echo "Building front image..."
    docker build -t "$FRONT_IMAGE" pull_simulator_front
}

stop() {
    docker rm -f "$FRONT_CONTAINER" "$BACKEND_CONTAINER" >/dev/null 2>&1 || true
}

run() {
    if ! docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
        echo "Error: '$DB_CONTAINER' is not running. Start it with pull_simulator/script/dev/docker-db.sh" >&2
        exit 1
    fi

    # One network for the three containers, so they reach each other by name.
    docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
    if ! docker inspect -f '{{json .NetworkSettings.Networks}}' "$DB_CONTAINER" | grep -q "\"$NETWORK\""; then
        docker network connect "$NETWORK" "$DB_CONTAINER"
    fi

    stop

    echo "Starting backend..."
    docker run -d --name "$BACKEND_CONTAINER" --network "$NETWORK" \
        -e DB_HOST="$DB_CONTAINER" -e DB_PORT=3306 \
        -e CORS_ALLOWED_ORIGINS="http://localhost:$FRONT_PORT" \
        -e SKIP_ASSETS="${SKIP_ASSETS:-0}" \
        -v "$STATIC_VOLUME":/app/static \
        -p "$BACKEND_PORT":8000 "$BACKEND_IMAGE" >/dev/null

    echo "Waiting for the backend (migrations, first-run picture download)..."
    until curl -sf "http://localhost:$BACKEND_PORT/api/banners" >/dev/null 2>&1; do
        if [ "$(docker inspect -f '{{.State.Running}}' "$BACKEND_CONTAINER" 2>/dev/null)" != "true" ]; then
            echo "Error: the backend stopped. Logs:" >&2
            docker logs --tail 50 "$BACKEND_CONTAINER" >&2
            exit 1
        fi
        sleep 2
    done

    # Test login traveler@test.local / 123456 (needs the tables, so after the backend migrated).
    ./pull_simulator/script/dev/seed-test-users.sh >/dev/null

    echo "Starting front..."
    docker run -d --name "$FRONT_CONTAINER" --network "$NETWORK" \
        -p "$FRONT_PORT":80 "$FRONT_IMAGE" >/dev/null

    echo
    echo "Front:   http://localhost:$FRONT_PORT   (login traveler@test.local / 123456)"
    echo "Backend: http://localhost:$BACKEND_PORT/docs"
    echo "Logs:    docker logs -f $BACKEND_CONTAINER   |   Stop: $0 stop"
}

case "${1:-all}" in
    build) build ;;
    run)   run ;;
    stop)  stop ;;
    all)   build && run ;;
    *)     echo "Usage: $0 [build|run|stop]"; exit 1 ;;
esac
