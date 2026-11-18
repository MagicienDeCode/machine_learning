# FastAPI backend. Built and started by ../build_and_run.sh.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml ./
RUN mkdir app && touch app/__init__.py && pip install . && rm -rf app build *.egg-info

COPY alembic.ini ./
COPY app ./app
COPY migrations ./migrations
COPY data ./data
COPY docker_entrypoint.sh ./
RUN chmod +x docker_entrypoint.sh && mkdir -p static

EXPOSE 8000
ENTRYPOINT ["./docker_entrypoint.sh"]
