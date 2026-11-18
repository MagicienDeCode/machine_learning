#!/bin/sh
# Build the Tetris AI Docker image and run it. The game window opens
# automatically in macOS Screen Sharing (no browser, nothing to install).
#   ./build_and_run.sh          build + run
#   ./build_and_run.sh build    only build
#   ./build_and_run.sh run      only run
set -e
cd "$(dirname "$0")"

IMAGE=tetris-ai
CONTAINER=tetris-ai
VNC_PORT=5901          # Mac-side port (5900 may be used by macOS itself)
VNC_PASSWORD=tetris

build() {
    docker build --platform linux/amd64 -t "$IMAGE" .
}

run() {
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
    docker run -d --name "$CONTAINER" --platform linux/amd64 \
        -e VNC_PASSWORD="$VNC_PASSWORD" -p "$VNC_PORT":5900 "$IMAGE" >/dev/null

    # Stop the container if you press Ctrl+C
    trap 'echo; echo "Stopping..."; docker rm -f "$CONTAINER" >/dev/null 2>&1; exit 0' INT TERM

    echo "Waiting for the game screen..."
    # Wait until the VNC server inside the container answers with its "RFB" greeting
    until [ "$( (sleep 1) | nc -w 2 localhost "$VNC_PORT" 2>/dev/null | head -c 3)" = "RFB" ]; do
        sleep 0.5
    done
    open "vnc://:$VNC_PASSWORD@localhost:$VNC_PORT"
    echo "(If Screen Sharing asks for a password, enter: $VNC_PASSWORD)"

    # Show game output until it finishes
    docker logs -f "$CONTAINER"
    docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}

case "${1:-all}" in
    build) build ;;
    run)   run ;;
    all)   build && run ;;
    *)     echo "Usage: $0 [build|run]"; exit 1 ;;
esac
