FROM ghcr.io/astral-sh/uv:python3.15-bookworm-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NEXUS_WEB_SERVER=uvicorn \
    NEXUS_HOST=0.0.0.0 \
    AUTO_LOGIN=false \
    NEXUS_ALLOW_REAL_WEB_TRADING=false

COPY pyproject.toml README.md ./
COPY nexus ./nexus

RUN uv sync --no-dev --extra ai --no-install-project
COPY . .
RUN uv sync --no-dev --extra ai

EXPOSE 8000

CMD ["sh", "-c", "uv run python -m nexus.main --web --demo --host 0.0.0.0 --port ${PORT:-8000}"]
