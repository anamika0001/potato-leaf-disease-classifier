"""
model_service.py — Loads the saved Keras CNN model and provides inference.

Key design decisions:
  • The model is loaded ONCE at import time (called from main.py on startup).
  • The Rescaling(1./255) layer is baked into the saved model, so the caller
    must pass pixel values on the 0-255 scale — no additional normalisation.
  • Inference uses `training=False` so BatchNorm uses running statistics
    and Dropout is inactive.
"""

from pathlib import Path
import logging
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------
_model = None          # holds the loaded Keras model (or None on failure)
_class_names: list[str] = []  # ordered class labels

# Resolve paths relative to this file → project root → models/
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _PROJECT_ROOT / "models" / "potato_cnn_model.keras"
_CLASS_NAMES_PATH = _PROJECT_ROOT / "models" / "class_names.txt"

# Safety limits
IMG_SIZE = (128, 128)                 # expected input dimensions
MAX_IMAGE_BYTES = 5 * 1024 * 1024     # 5 MB upload limit
MAX_IMAGE_PIXELS = 20_000_000         # decompression-bomb guard (~20 MP)


def load_model() -> bool:
    """
    Load the saved Keras model and class-name list.
    Returns True on success, False otherwise.
    Called once from main.py's lifespan handler.
    """
    global _model, _class_names

    # --- Load class names ---------------------------------------------------
    if not _CLASS_NAMES_PATH.exists():
        logger.error("Class-name file not found: %s", _CLASS_NAMES_PATH)
        return False

    _class_names = [
        line.strip()
        for line in _CLASS_NAMES_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(_class_names) != 3:
        logger.error("Expected 3 class names, found %d", len(_class_names))
        return False

    # --- Load Keras model ---------------------------------------------------
    if not _MODEL_PATH.exists():
        logger.error("Model file not found: %s", _MODEL_PATH)
        return False

    try:
        import tensorflow as tf
        _model = tf.keras.models.load_model(str(_MODEL_PATH))
        logger.info("Model loaded successfully from %s", _MODEL_PATH)
        logger.info("Class names: %s", _class_names)
        return True
    except Exception:
        logger.exception("Failed to load model")
        _model = None
        return False


def is_model_loaded() -> bool:
    """Check whether the model is ready for inference."""
    return _model is not None


def get_class_names() -> list[str]:
    """Return the ordered list of class names."""
    return list(_class_names)


def predict(image_array: np.ndarray) -> dict:
    """
    Run inference on a preprocessed image array.

    Parameters
    ----------
    image_array : np.ndarray
        Shape (1, 128, 128, 3), dtype float32, pixel values 0-255.

    Returns
    -------
    dict  with keys: predicted_class, display_name, confidence, probabilities
    """
    if _model is None:
        raise RuntimeError("Model is not loaded")

    # Predict with training=False (BatchNorm & Dropout inactive)
    scores = _model(image_array, training=False)   # returns a tf.Tensor
    scores = scores.numpy()[0]                      # shape (3,)

    best_idx = int(np.argmax(scores))
    predicted_class = _class_names[best_idx]

    # Build a human-friendly display name
    display_name = predicted_class.replace("Potato___", "").replace("_", " ")

    probabilities = {
        name: round(float(scores[i]), 6)
        for i, name in enumerate(_class_names)
    }

    return {
        "predicted_class": predicted_class,
        "display_name": display_name,
        "confidence": round(float(scores[best_idx]), 6),
        "probabilities": probabilities,
    }
