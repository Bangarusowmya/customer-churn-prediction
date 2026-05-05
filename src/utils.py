import os
import logging
import joblib

def get_logger(name: str):
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"[✓] Model saved → {path}")

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at: {path}")
    model = joblib.load(path)
    print(f"[✓] Model loaded ← {path}")
    return model

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)