FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    UV_CACHE_DIR=/root/.cache/uv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    ffmpeg \
 && rm -rf /var/lib/apt/lists/*

 # Install UV
COPY --from=ghcr.io/astral-sh/uv:0.9.2 /uv /uvx /bin/

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-cache --no-dev --no-install-project

# Preload deep learning models
RUN uv run python -c "import whisper; whisper.load_model('turbo')"
RUN uv run hf download badrex/mms-300m-arabic-dialect-identifier

# Copy the rest of the application
COPY . .

# Install the project
RUN uv sync --frozen --no-cache --no-dev

# Start the application
EXPOSE 8000
CMD ["uv", "run", "babel", "--server.port=8000", "--server.address=0.0.0.0", "--browser.gatherUsageStats=false"]
