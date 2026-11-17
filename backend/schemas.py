"""
schemas.py — Pydantic models for FastAPI request/response validation.
"""

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response schema for the /health endpoint."""
    status: str              # "ready" or "unavailable"
    model_loaded: bool       # True when the CNN has been loaded successfully


class PredictionResponse(BaseModel):
    """Response schema for the /predict endpoint."""
    predicted_class: str          # e.g. "Potato___Early_blight"
    display_name: str             # e.g. "Early Blight"
    confidence: float             # softmax score for the predicted class
    probabilities: dict[str, float]  # all three class scores
