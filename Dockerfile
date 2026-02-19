# Stage 1: Build the React app
FROM node:20-alpine AS node-base
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY frontend/ .
RUN REACT_APP_BACKEND_PORT=8000 pnpm run build

# Stage 2: Install Python dependencies with UV
FROM python:3.12-slim AS builder-base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock .python-version ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --extra tf

# Stage 3: Production image
FROM python:3.12-slim AS production
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV FASTAPI_ENV=production
COPY --from=builder-base /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Preload the model and dataset
RUN python3 -m nltk.downloader wordnet
RUN python3 -c "import tensorflow as tf; tf.keras.applications.VGG16(weights='imagenet')"
RUN python3 -c "import tensorflow_datasets as tfds; tfds.load('imagenette/320px-v2', shuffle_files=False, with_info=True, as_supervised=True, batch_size=None)"

COPY --from=builder-base /app/src/ ./src/
COPY --from=node-base /app/build ./src/channelexplorer/static
COPY examples/ ./examples/

EXPOSE 8000

CMD redis-server --daemonize yes && python3 examples/run_tf.py
