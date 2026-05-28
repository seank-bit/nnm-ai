ARG DEVICE=cpu

FROM python:3.12-slim-bookworm AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    default-jre-headless libpq5 libpq-dev curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv==0.5.4
RUN uv venv "$VIRTUAL_ENV"

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY alembic.ini ./

FROM base AS build-cpu
RUN uv pip install -e ".[cpu]"

FROM base AS build-gpu
RUN uv pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.5" \
 && uv pip install -e ".[gpu]"

FROM build-${DEVICE} AS final
RUN mkdir -p /app/var/extracted /app/var/colbert
EXPOSE 80
CMD ["uvicorn", "nnm.main:app", "--host", "0.0.0.0", "--port", "80"]
