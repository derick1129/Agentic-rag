FROM ghcr.io/astral-sh/uv:0.8.17-python3.11-bookworm-slim AS builder

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

COPY . .
RUN uv sync --frozen --no-dev

FROM builder AS test
ENV UV_NO_DEV=0
RUN uv sync --frozen --group dev
ENTRYPOINT ["uv", "run", "--frozen", "--group", "dev", "pytest"]

FROM builder AS evaluator-builder
RUN uv sync --frozen --no-dev --extra phoenix-eval

FROM python:3.11-slim-bookworm AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app /tmp/adaptive_rag_ingestion \
    && chown -R app:app /app /tmp/adaptive_rag_ingestion

WORKDIR /app
COPY --from=builder --chown=app:app /app /app

USER app
ENTRYPOINT ["/app/docker/entrypoint.sh"]

FROM python:3.11-slim-bookworm AS runtime-eval

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system app && useradd --system --gid app --create-home app \
    && mkdir -p /app /tmp/adaptive_rag_ingestion \
    && chown -R app:app /app /tmp/adaptive_rag_ingestion

WORKDIR /app
COPY --from=evaluator-builder --chown=app:app /app /app

USER app
ENTRYPOINT ["/app/docker/entrypoint.sh"]
