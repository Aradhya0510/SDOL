"""Structured logger for Provena."""

from __future__ import annotations

import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Get a structured logger for Provena components."""
    logger = logging.getLogger(f"provena.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
