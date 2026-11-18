#!/bin/sh
# Runs INSIDE the container (do not run it on your Mac - use ./build_and_run.sh).
# Starts a virtual screen + VNC server, then plays the game.
set -e

if [ ! -f /.dockerenv ]; then
    echo "This script runs inside the Docker container. Use ./build_and_run.sh instead."
    exit 1
fi

# Virtual screen exactly the size of the game window (520x560)
Xvfb :99 -screen 0 520x560x24 >/dev/null 2>&1 &
sleep 1
x11vnc -display :99 -forever -shared -passwd "${VNC_PASSWORD:-snake}" -quiet -rfbport 5900 >/dev/null 2>&1 &

i=${START_DELAY:-5}
while [ "$i" -gt 0 ]; do
    echo "Game starts in $i s ..."
    sleep 1
    i=$((i - 1))
done

# Hide harmless torch "NNPACK unsupported hardware" warnings
python test_cnn.py 2>&1 | grep --line-buffered -v NNPACK
