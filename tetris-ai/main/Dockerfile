# Tetris AI - play the trained MaskablePPO model inside Docker.
# The game window is drawn on a virtual X display (Xvfb) and served over VNC;
# build_and_run.sh opens it in macOS Screen Sharing automatically.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DISPLAY=:99 \
    SDL_AUDIODRIVER=dummy \
    PYGAME_HIDE_SUPPORT_PROMPT=1 \
    SDL_VIDEO_WINDOW_POS=0,0 \
    START_DELAY=5

# Virtual display + VNC server, and fonts for pygame
RUN apt-get update && apt-get install -y --no-install-recommends \
        xvfb x11vnc \
        fontconfig fonts-dejavu-core \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Same package versions the model was trained with (CPU-only torch)
RUN pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install \
        numpy==2.2.1 \
        gymnasium==1.2.1 \
        pygame==2.6.1 \
        stable-baselines3==2.7.0 \
        sb3-contrib==2.7.0 \
        opencv-python-headless==4.12.0.88

WORKDIR /app

# Game code and assets
COPY tetris_game.py tetris_game_cnn_wrapper.py test_cnn.py ./
COPY sounds/ sounds/

# No MPS inside Docker, so test_cnn.py looks in trained_models_cnn/ -> put the model there
COPY trained_models_cnn_mps/ppo_tetris_final.zip trained_models_cnn/ppo_tetris_final.zip

COPY docker_entrypoint.sh /usr/local/bin/docker_entrypoint.sh
RUN chmod +x /usr/local/bin/docker_entrypoint.sh

EXPOSE 5900
ENTRYPOINT ["/usr/local/bin/docker_entrypoint.sh"]
