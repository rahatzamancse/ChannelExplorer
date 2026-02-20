# Stage 1: Build the Next.js frontend
FROM node:20-alpine AS frontend
WORKDIR /app
COPY frontend/package.json frontend/pnpm-lock.yaml ./
RUN corepack enable && pnpm install --frozen-lockfile
COPY frontend/ .
ENV NEXT_PUBLIC_API_URL=/api
RUN pnpm run build

# Stage 2: Install Python dependencies with uv
FROM python:3.12-slim AS builder
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src/ ./src/

RUN uv sync --frozen --no-dev --extra tf

# Stage 3: Production image
FROM python:3.12-slim AS production
RUN apt-get update && apt-get install -y redis-server && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV FASTAPI_ENV=production
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Preload the InceptionV3 model and Imagenette dataset
RUN /app/.venv/bin/python -c "import tensorflow as tf; tf.keras.applications.InceptionV3(weights='imagenet')"
RUN /app/.venv/bin/python -c "import tensorflow_datasets as tfds; tfds.load('imagenette/320px-v2', shuffle_files=False, with_info=True, as_supervised=True, batch_size=None)"

COPY --from=builder /app/src/ ./src/
COPY --from=frontend /app/out ./src/channelexplorer/static
COPY examples/docker_demo.py ./examples/docker_demo.py

EXPOSE 8000

CMD ["sh", "-c", "redis-server --daemonize yes && python3 examples/docker_demo.py"]
