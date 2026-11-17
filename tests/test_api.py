"""
test_api.py — Automated tests for the FastAPI backend.

Run with:  python -m pytest tests/test_api.py -v
"""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Ensure project root is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.main import app  # noqa: E402

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_test_image(fmt: str = "JPEG", size=(128, 128)) -> bytes:
    """Create a small in-memory image and return its bytes."""
    img = Image.new("RGB", size, color=(34, 139, 34))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


FAKE_SCORES = np.array([[0.85, 0.10, 0.05]], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Health-check endpoint
# ---------------------------------------------------------------------------
def test_health_endpoint():
    """GET /health should return 200 with status and model_loaded keys."""
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert "status" in body
    assert "model_loaded" in body


# ---------------------------------------------------------------------------
# 2. Successful prediction (mocked model)
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
@patch("backend.main.predict")
def test_valid_image_prediction(mock_predict, mock_loaded):
    """POST /predict with a valid JPEG should return 200 + correct schema."""
    mock_predict.return_value = {
        "predicted_class": "Potato___Early_blight",
        "display_name": "Early Blight",
        "confidence": 0.85,
        "probabilities": {
            "Potato___Early_blight": 0.85,
            "Potato___Late_blight": 0.10,
            "Potato___healthy": 0.05,
        },
    }
    img_bytes = _make_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_class"] == "Potato___Early_blight"
    assert body["display_name"] == "Early Blight"
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["probabilities"]) == 3


# ---------------------------------------------------------------------------
# 3. Corrupted image → 400
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
def test_corrupted_image_rejected(mock_loaded):
    """POST /predict with garbage bytes should return 400."""
    resp = client.post(
        "/predict",
        files={"file": ("bad.jpg", b"not_an_image", "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "not a valid image" in resp.json()["detail"].lower() or "corrupted" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 4. Unsupported file format → 400
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
def test_unsupported_format_rejected(mock_loaded):
    """POST /predict with a .gif extension should return 400."""
    img_bytes = _make_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.gif", img_bytes, "image/gif")},
    )
    assert resp.status_code == 400
    assert "unsupported" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 5. Oversized file → 400
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
def test_oversized_file_rejected(mock_loaded):
    """POST /predict with >5 MB should return 400."""
    # Create oversized bytes: valid JPEG header + padding
    img_bytes = _make_test_image()
    big_bytes = img_bytes + b"\x00" * (6 * 1024 * 1024)
    resp = client.post(
        "/predict",
        files={"file": ("big.jpg", big_bytes, "image/jpeg")},
    )
    assert resp.status_code == 400
    assert "too large" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# 6. Prediction output structure
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
@patch("backend.main.predict")
def test_prediction_output_structure(mock_predict, mock_loaded):
    """The response must contain the four expected keys."""
    mock_predict.return_value = {
        "predicted_class": "Potato___healthy",
        "display_name": "healthy",
        "confidence": 0.95,
        "probabilities": {
            "Potato___Early_blight": 0.03,
            "Potato___Late_blight": 0.02,
            "Potato___healthy": 0.95,
        },
    }
    img_bytes = _make_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.png", img_bytes, "image/png")},
    )
    body = resp.json()
    assert set(body.keys()) == {"predicted_class", "display_name", "confidence", "probabilities"}


# ---------------------------------------------------------------------------
# 7. Correct class order
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=True)
@patch("backend.main.predict")
def test_correct_class_order(mock_predict, mock_loaded):
    """The probabilities dict must contain exactly the three expected classes."""
    expected_classes = {"Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"}
    mock_predict.return_value = {
        "predicted_class": "Potato___Late_blight",
        "display_name": "Late Blight",
        "confidence": 0.78,
        "probabilities": {
            "Potato___Early_blight": 0.12,
            "Potato___Late_blight": 0.78,
            "Potato___healthy": 0.10,
        },
    }
    img_bytes = _make_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    body = resp.json()
    assert set(body["probabilities"].keys()) == expected_classes


# ---------------------------------------------------------------------------
# 8. Model not loaded → 503
# ---------------------------------------------------------------------------
@patch("backend.main.is_model_loaded", return_value=False)
def test_model_not_loaded_returns_503(mock_loaded):
    """When the model is unavailable, /predict should return 503."""
    img_bytes = _make_test_image()
    resp = client.post(
        "/predict",
        files={"file": ("leaf.jpg", img_bytes, "image/jpeg")},
    )
    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()
