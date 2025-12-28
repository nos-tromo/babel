import sys
from pathlib import Path

import whisper  # type: ignore

from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from loguru import logger

from babel.utils.env_cfg import load_model_env, load_path_env
from babel.utils.logging_cfg import setup_logging

load_dotenv()
setup_logging()


def resolve_model_path(model_name: str, cache_folder: Path) -> str:
    """
    Resolves the model path to the local cache directory if available.
    This helps bypass online checks in the transformers library when running offline.

    Args:
        model_name (str): The name of the model (e.g., "bert-base-uncased").
        cache_folder (Path): The path to the Hugging Face cache directory.

    Returns:
        str: Local path to the model if cached, else the original model name.
    """
    if "/" in model_name and not Path(model_name).exists():
        repo_id = model_name
        model_dir_name = f"models--{repo_id.replace('/', '--')}"
        model_cache_dir = cache_folder / model_dir_name

        if model_cache_dir.exists():
            ref_path = model_cache_dir / "refs" / "main"
            if ref_path.exists():
                with open(ref_path, "r") as f:
                    commit_hash = f.read().strip()
                snapshot_path = model_cache_dir / "snapshots" / commit_hash
                if snapshot_path.exists():
                    return str(snapshot_path)
    return model_name


def load_hf_model(model_id: str, cache_folder: Path) -> None:
    """
    Loads and returns the HuggingFace embedding model.

    Args:
        model_id (str): The name of the model to load.
        cache_folder (Path): The path to the cache folder.
    """
    resolved_model_name = resolve_model_path(model_id, cache_folder)

    if resolved_model_name != model_id:
        logger.info("Found local cache for {} at {}", model_id, resolved_model_name)
        model_id = resolved_model_name
    else:
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_folder,
        )

    logger.info("Loaded model: {}", model_id)


def load_whisper_model(model_id: str) -> None:
    """
    Loads and returns the Whisper model.

    Args:
        model_id (str): The name of the model to load.
    """
    whisper.load_model("turbo")
    logger.info("Loaded whisper model: {}", model_id)


def main() -> None:
    """
    Main function to verify configuration loading.
    """
    # Load configurations
    paths = load_path_env()
    models = load_model_env()

    # Log the loaded configurations
    for path in paths.__dataclass_fields__.keys():
        logger.info("{} path: {}", path, getattr(paths, path))
    for model in models.__dataclass_fields__.keys():
        logger.info("{}: {}", model, getattr(models, model))

    # Load the app's models
    # Hugging Face
    load_hf_model(
        model_id=models.dialect_model,
        cache_folder=paths.hf_hub_cache,
    )

    # Whisper
    load_whisper_model(models.whisper_model)

    logger.info("All models loaded successfully.")


if __name__ == "__main__":
    sys.path.append(str(Path(__file__).parents[2].resolve()))
    main()
