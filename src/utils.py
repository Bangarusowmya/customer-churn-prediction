"""
utils.py - Shared helper functions used across the pipeline.

Nothing fancy here, just stuff I kept copy-pasting so I moved it out.
"""

import os
import logging
import pickle
from datetime import datetime


def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to console with a clean format.
    Using this instead of print() everywhere so we get timestamps.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                datefmt="%Y-%m-%d %H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger


def save_model(model, path: str) -> None:
    """Pickle the trained model to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"[✓] Model saved → {path}")


def load_model(path: str):
    """Load a pickled model from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"[✓] Model loaded ← {path}")
    return model


def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist. Useful for output paths."""
    os.makedirs(path, exist_ok=True)
