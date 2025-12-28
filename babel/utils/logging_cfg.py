from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger

from babel.utils.env_cfg import load_path_env

LOG_PATH = load_path_env().logs


def setup_logging(
    log_path: Path | None = None,
    encoding="utf-8",
    rotation: str = "5 MB",
    retention: int = 3,
    backtrace: bool = False,
    diagnose: bool = False,
) -> Path:
    """
    Set up logging for the application.

    Args:
        log_path (Path | None, optional): The path to the log file. If None, uses default from env. Defaults to None.
        encoding (str, optional): The log file encoding. Defaults to "utf-8".
        rotation (str, optional): The log file rotation policy. Defaults to "5 MB".
        retention (int, optional): The number of log files to retain. Defaults to 3
        backtrace (bool, optional): Whether to include backtrace information. Defaults to False.
        diagnose (bool, optional): Whether to include diagnostic information. Defaults to False.

    Returns:
        Path: The path to the log file.
    """
    log_path = LOG_PATH if log_path is None else log_path

    logger.remove()

    logger.add(
        sink=sys.stderr,
        level="INFO",
        backtrace=backtrace,
        diagnose=diagnose,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name} | {message}",
    )

    logger.add(
        sink=log_path,
        rotation=rotation,
        retention=retention,
        encoding=encoding,
        level="DEBUG",
        backtrace=True,
        diagnose=diagnose,
        enqueue=True,
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {line:<4} | {name} | {message}",
    )

    return log_path
