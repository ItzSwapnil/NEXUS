# syntax=docker/dockerfile:1
FROM python:3.13-slim AS base

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.local/bin:$PATH"' >> /etc/profile
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app
COPY pyproject.toml uv.lock /app/

# Sync dependencies (no dev)
RUN uv sync --frozen

# Copy source
COPY . /app

# Master panel (no web server) – use auto-demo by default
CMD ["uv", "run", "python", "-m", "nexus.main", "--auto-demo"]
