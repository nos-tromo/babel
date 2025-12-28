# Babel

Babel is a Streamlit application that uses deep learning models to identify Arabic dialects from audio and video recordings. It leverages OpenAI's Whisper for language detection and specialized transformer models for dialect classification.

## Features

- **Multi-format Support**: Upload audio or video files in MP3, M4A, WAV, OGG, FLAC, MP4, MKV, AVI, MOV, or WEBM formats.
- **Audio Slicing**: Analyze specific segments of your media by specifying start time and duration.
- **Language Detection**: Automatically verifies if the spoken language is Arabic using OpenAI Whisper.
- **Dialect Identification**: Classifies the specific Arabic dialect with confidence scores.
- **Hardware Acceleration**: Automatically utilizes CUDA (NVIDIA) or MPS (Apple Silicon) if available.

## Prerequisites

- Python 3.11 or higher
- [ffmpeg](https://ffmpeg.org/) (required for audio processing)

## Installation

This project uses `uv` for dependency management.

1. **Clone the repository:**

   ```bash
   git clone https://github.com/nos-tromo/babel.git
   cd babel
   ```

2. **Install dependencies:**

   ```bash
   uv sync
   ```

   Or using pip:

   ```bash
   pip install .
   ```

## Usage

Start locally (uses default host/port):

```bash
uv run babel
```

Bind to all interfaces or change port if needed:

```bash
uv run babel --server.address=0.0.0.0 --server.port=8000 --browser.gatherUsageStats=false
```

If installed as a package, the `babel` console script is available and accepts the same flags.

## Docker

### Single container

1. **Build**

   ```bash
   docker build -t babel .
   ```

2. **Run (CPU-only example)**

   ```bash
   docker run \
     -p 8000:8000 \
     -e LOG_PATH=/app/.logs/babel.log \
     -v babel-logs:/app/.logs \
     -v model-cache:/root/.cache \
     babel
   ```

   For NVIDIA GPUs, add `--gpus all`.

### Docker Compose (recommended)

Profiles are defined for CPU and GPU. The image uses persistent volumes for logs and model cache so you don’t redownload models on rebuilds.

- CPU: `docker compose --profile cpu up --build`
- GPU: `docker compose --profile gpu up --build` (requires NVIDIA runtime)

Service ports: `8000` exposed locally. Logs are written to `babel-logs` volume; model weights/cache stored in `model-cache`.

## Configuration

You can configure the models used by setting environment variables in a `.env` file:

```env
DIALECT_MODEL=badrex/mms-300m-arabic-dialect-identifier
WHISPER_MODEL=turbo
```

- `DIALECT_MODEL`: The Hugging Face model ID for dialect identification.
- `WHISPER_MODEL`: The Whisper model size (e.g., `tiny`, `base`, `small`, `medium`, `large`, `turbo`).

For Docker configurations, populate an `.env.docker` file in the project's root:

```bash
cp .env.docker.example .env.docker
```

## Development

### Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality.

```bash
uv run pre-commit install
```

### Running Tests

```bash
uv run pytest
```
