ARG DEVICE=cpu

FROM python:3.12-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jre-headless libpq5 curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv==0.5.4
WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY alembic.ini ./

FROM base AS build-cpu
RUN uv pip install --system -e ".[cpu]"

FROM base AS build-gpu
RUN uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.5" \
 && uv pip install --system -e ".[gpu]"

FROM build-${DEVICE} AS final
RUN mkdir -p /app/var/extracted /app/var/colbert
EXPOSE 8000
CMD ["uvicorn", "nnm.main:app", "--host", "0.0.0.0", "--port", "8000"]
