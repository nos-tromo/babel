FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-cache --no-dev --no-install-project

COPY . .

RUN uv sync --frozen --no-cache --no-dev

EXPOSE 8000

CMD ["sh", "-c", "uv run load-models && uv run babel --server.address=0.0.0.0 --server.port=8000 --browser.gatherUsageStats=false"]
