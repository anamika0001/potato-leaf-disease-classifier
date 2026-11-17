"""
main.py — FastAPI backend for the Potato Leaf Disease Detector.

Endpoints
---------
GET  /health   → API readiness & model-load status
POST /predict  → accepts an image upload, returns disease prediction JSON
"""

import io
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError

from backend.model_service import (
    IMG_SIZE,
    MAX_IMAGE_BYTES,
    MAX_IMAGE_PIXELS,
    get_class_names,
    is_model_loaded,
    load_model,
    predict,
)
from backend.schemas import HealthResponse, PredictionResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the CNN model when the server starts."""
    success = load_model()
    if success:
        logger.info("✅ Model ready for inference")
    else:
        logger.warning("⚠️  Model could NOT be loaded — /predict will return 503")
    yield  # server is now running
    logger.info("Server shutting down")


# ---------------------------------------------------------------------------
# App instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Potato Leaf Disease Detector API",
    description="Predicts potato leaf diseases using a trained CNN model.",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow requests from the Streamlit frontend (default localhost:8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",
        "http://127.0.0.1:8501",
    ],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Supported upload types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/jpg",
}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Return API availability and model-loading status."""
    loaded = is_model_loaded()
    return HealthResponse(
        status="ready" if loaded else "unavailable",
        model_loaded=loaded,
    )


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict_disease(file: UploadFile = File(...)):
    """
    Accept an uploaded leaf image and return the CNN prediction.

    Validations:
      1. File extension must be .jpg / .jpeg / .png
      2. File size must be ≤ 5 MB
      3. File bytes must decode to a valid image
      4. Image pixel count must be below the decompression-bomb limit
    """
    # 0. Model must be available
    if not is_model_loaded():
        raise HTTPException(
            status_code=503,
            detail="Prediction service is temporarily unavailable. The model could not be loaded.",
        )

    # 1. Validate extension
    filename = file.filename or ""
    ext = ""
    if "." in filename:
        ext = "." + filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Please upload a JPG, JPEG, or PNG image.",
        )

    # 2. Read bytes and check size
    try:
        contents = await file.read()
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read the uploaded file.")

    if len(contents) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=400,
            detail=f"File is too large ({len(contents) / (1024*1024):.1f} MB). Maximum allowed size is 5 MB.",
        )

    # 3. Decode image
    try:
        pil_image = Image.open(io.BytesIO(contents))
        pil_image.verify()  # check header integrity
        # Re-open after verify (verify closes the fp)
        pil_image = Image.open(io.BytesIO(contents))
    except (UnidentifiedImageError, Exception):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image or is corrupted.",
        )

    # 4. Decompression-bomb guard
    w, h = pil_image.size
    if w * h > MAX_IMAGE_PIXELS:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions ({w}×{h}) exceed the safe limit. Please use a smaller image.",
        )

    # 5. Preprocess — match the notebook's inference cell exactly
    pil_image = pil_image.convert("RGB")
    pil_image = pil_image.resize(IMG_SIZE, Image.LANCZOS)
    image_array = np.array(pil_image, dtype=np.float32)        # (128,128,3), 0-255
    image_array = np.expand_dims(image_array, axis=0)          # (1,128,128,3)
    # NOTE: DO NOT rescale to 0-1 — the model's Rescaling layer does that.

    # 6. Run prediction
    try:
        result = predict(image_array)
    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="An error occurred during prediction. Please try again.",
        )

    return PredictionResponse(**result)
