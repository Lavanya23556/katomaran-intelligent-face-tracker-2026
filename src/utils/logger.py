"""
Structured event logging for VisiTrace.
Maintains the mandatory events.log file and formats console logs.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    log_file_path: str = "events.log",
    level: int = logging.INFO
) -> logging.Logger:
    """
    Configure and return the root VisiTrace logger.
    Logs to both stderr and the mandatory events.log file.
    """
    logger = logging.getLogger("visitrace")
    logger.setLevel(level)
    logger.propagate = False

    # Clear existing handlers if re-called
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formatter with UTC-like timestamp, level, and message
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler for mandatory events.log
    log_path = Path(log_file_path)
    if str(log_path.parent) not in ("", "."):
        log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)
    logger.addHandler(file_handler)

    return logger


# Global logger instance
logger = logging.getLogger("visitrace")
if not logger.handlers:
    setup_logger()
