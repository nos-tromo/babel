# Babel

Babel is a Streamlit application that uses deep learning models to identify Arabic dialects from audio and video recordings. It leverages OpenAI's Whisper for language detection and specialized transformer models for dialect classification.

## Features

- **Multi-format Support**: Upload audio or video files in MP3, WAV, OGG, FLAC, MP4, MKV, AVI, MOV, or WEBM formats.
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

To start the application, run:

```bash
uv run babel
```

Or if installed in your environment:

```bash
babel
```

This will launch the Streamlit interface in your default web browser.

## Docker

You can also run Babel using Docker.

1. **Build the Docker image:**

    ```bash
    docker build -t babel .
    ```

2. **Run the container:**

    ```bash
    docker run -p 8000:8000 --gpus all babel
    ```

    The application will be available at `http://localhost:8000`.

    > **Note:** The `--gpus all` flag is required for NVIDIA GPU support. For Apple Silicon or CPU-only usage, you can omit it, but performance may be slower.

## Configuration

You can configure the models used by setting environment variables in a `.env` file:

```env
DIALECT_MODEL=badrex/mms-300m-arabic-dialect-identifier
WHISPER_MODEL=turbo
```

- `DIALECT_MODEL`: The Hugging Face model ID for dialect identification.
- `WHISPER_MODEL`: The Whisper model size (e.g., `tiny`, `base`, `small`, `medium`, `large`, `turbo`).

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
