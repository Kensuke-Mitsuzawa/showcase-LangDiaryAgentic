FROM python:3.11-slim

# Install uv directly
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# We add g++, python3-dev, and build-essential
RUN apt-get update && apt-get install -y \
    curl \
    g++ \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*
# -----------------------

# Copy only dependency files first (to leverage Docker layer caching)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies
# RUN pip install uv
# COPY pyproject.toml .
# COPY uv.lock .
# RUN uv sync
# Install dependencies
RUN uv sync --frozen --no-cache

# Copy the rest of the application code
COPY . .

# Ensure the uv-created binaries are in the PATH
ENV PATH="/app/.venv/bin:$PATH"