import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()


@dataclass(frozen=True)
class ModelConfig:
    """
    Dataclass for model configuration.
    """

    dialect_model: str
    whisper_model: str


@dataclass(frozen=True)
class PathConfig:
    """
    Dataclass for path configuration.
    """

    hf_hub_cache: Path
    logs: Path


def load_model_env() -> ModelConfig:
    """
    Loads model configuration from environment variables or defaults.

    Returns:
        ModelConfig: Dataclass containing model configuration.
        - dialect_model (str): The dialect model identifier.
        - whisper_model (str): The Whisper model identifier.
    """
    default_dialect_model = "badrex/mms-300m-arabic-dialect-identifier"
    default_whisper_model = "turbo"

    return ModelConfig(
        dialect_model=os.getenv("DIALECT_MODEL", default_dialect_model),
        whisper_model=os.getenv("WHISPER_MODEL", default_whisper_model),
    )


def load_path_env() -> PathConfig:
    """
    Loads path configuration from environment variables or defaults.

    Returns:
        PathConfig: Dataclass containing path configuration.
        - hf_hub_cache (Path): Path to the Hugging Face Hub cache directory.
        - logs (Path): Path to the logs file.
    """
    default_hf_hub_cache = Path.home() / ".cache" / "huggingface" / "hub"
    default_logs = Path(__file__).parents[2].resolve() / ".logs" / "babel.log"

    return PathConfig(
        hf_hub_cache=Path(os.getenv("HF_HUB_CACHE", default_hf_hub_cache)).expanduser(),
        logs=Path(os.getenv("LOG_PATH", default_logs)).expanduser(),
    )


def set_offline_env() -> None:
    """
    Sets environment variables to configure Hugging Face libraries for offline mode.

    This function ensures that Hugging Face libraries such as `transformers` and
    `llama_index` operate in offline mode by setting the appropriate environment
    variables. It should be invoked before importing these libraries to avoid
    unexpected behavior.

    Environment Variables Set:
    - `HF_HUB_OFFLINE`: Forces the Hugging Face Hub to operate in offline mode.
    - `TRANSFORMERS_OFFLINE`: Disables online access for the `transformers` library.
    - `HF_HUB_DISABLE_TELEMETRY`: Disables telemetry data collection by Hugging Face.
    - `HF_HUB_DISABLE_PROGRESS_BARS`: Suppresses progress bars in Hugging Face operations.
    - `HF_HUB_DISABLE_SYMLINKS_WARNING`: Disables symlink-related warnings.
    - `KMP_DUPLICATE_LIB_OK`: Resolves potential library duplication issues.

    Note:
    Call this function before importing `transformers` or `llama_index` to ensure
    the offline mode is applied correctly.
    """
    babel_offline = os.getenv("BABEL_OFFLINE", "1").lower() in {"1", "true", "yes"}

    if babel_offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        logger.info("Set Hugging Face libraries to offline mode.")
    else:
        logger.info("Hugging Face libraries are in online mode.")
