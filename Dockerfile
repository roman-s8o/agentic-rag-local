FROM python:3.12-slim AS base

COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --extra ml --extra ingest

COPY src ./src
COPY configs ./configs
RUN uv sync --frozen --extra ml --extra ingest

EXPOSE 8800
CMD ["uv", "run", "--no-sync", "rag", "serve"]
