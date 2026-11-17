"""
app.py — Streamlit frontend for the Potato Leaf Disease Detector.

This single-page app lets the user:
  1. Upload a potato leaf image (JPG / JPEG / PNG)
  2. Send it to the FastAPI backend for CNN inference
  3. View the prediction, confidence scores, and disease guidance
  4. Download a PDF report
  5. Reset and analyse another image

Run from the project root:
    python -m streamlit run frontend/app.py
"""

import os
import sys
from pathlib import Path

import requests
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import utils / backend info
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.disease_info import get_disease_info  # noqa: E402
from frontend.styles import get_custom_css          # noqa: E402
from utils.pdf_generator import generate_report     # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_BASE_URL = st.secrets.get(
    "API_BASE_URL",
    os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
)
API_TIMEOUT = 60  # seconds

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="centered",
)

# Inject custom CSS
st.markdown(get_custom_css(), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------
def _init_state():
    """Initialise session-state keys the first time the app loads."""
    defaults = {
        "prediction": None,       # dict returned by the API
        "uploaded_bytes": None,   # raw image bytes for the PDF report
        "uploaded_name": None,    # original filename
        "analysis_done": False,   # prevents duplicate API calls on rerun
        "uploader_key": 0,        # used to reset the file uploader widget
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


_init_state()


# ---------------------------------------------------------------------------
# Helper — call the FastAPI backend
# ---------------------------------------------------------------------------
def _call_predict_api(image_bytes: bytes, filename: str) -> dict | None:
    """
    POST the image to /predict and return the JSON response dict.
    Returns None on any error (and shows a user-friendly message).
    """
    try:
        resp = requests.post(
            f"{API_BASE_URL}/predict",
            files={"file": (filename, image_bytes, "image/jpeg")},
            timeout=API_TIMEOUT,
        )
        if resp.status_code == 200:
            return resp.json()
        else:
            detail = resp.json().get("detail", resp.text)
            st.error(f"⚠️ Prediction failed: {detail}")
            return None

    except requests.ConnectionError:
        st.error(
            "🔌 **Unable to connect to the prediction service.**  \n"
            "Please check whether the FastAPI backend is running at "
            f"`{API_BASE_URL}`."
        )
        return None
    except requests.Timeout:
        st.error("⏳ The prediction request timed out. Please try again.")
        return None
    except Exception as exc:
        st.error(f"❌ An unexpected error occurred: {exc}")
        return None


# ---------------------------------------------------------------------------
# Helper — badge HTML
# ---------------------------------------------------------------------------
def _badge(display_name: str) -> str:
    """Return an HTML badge span coloured by disease type."""
    name_lower = display_name.lower()
    if "early" in name_lower:
        css_class = "badge-amber"
    elif "late" in name_lower:
        css_class = "badge-red"
    else:
        css_class = "badge-green"
    return f'<span class="{css_class}">{display_name}</span>'


# ===========================================================================
# MAIN UI
# ===========================================================================

# ---- Header ----------------------------------------------------------------
st.markdown('<h1 class="main-title">🥔 Potato Leaf Disease Detector</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">AI-powered analysis to identify common potato leaf diseases from images.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="instruction">Upload a clear image of a potato leaf to receive an AI-based prediction.</p>',
    unsafe_allow_html=True,
)

# ---- Upload section --------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose a leaf image",
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}",
    help="Supported formats: JPG, JPEG, PNG (max 5 MB)",
)

if uploaded_file is not None:
    # Show preview
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

    # Analyse button
    analyse_disabled = st.session_state.analysis_done
    if st.button("🔬 Analyze Leaf", type="primary", use_container_width=True, disabled=analyse_disabled):
        with st.spinner("Analysing image… please wait."):
            img_bytes = uploaded_file.getvalue()
            result = _call_predict_api(img_bytes, uploaded_file.name)

        if result is not None:
            st.session_state.prediction = result
            st.session_state.uploaded_bytes = uploaded_file.getvalue()
            st.session_state.uploaded_name = uploaded_file.name
            st.session_state.analysis_done = True
            st.rerun()

# ---- Results section -------------------------------------------------------
pred = st.session_state.prediction
if pred is not None:
    st.markdown("---")

    # Result card
    st.markdown('<div class="result-card">', unsafe_allow_html=True)

    display = pred["display_name"]
    confidence = pred["confidence"]
    st.markdown(f"### {_badge(display)} &nbsp; — &nbsp; Model Confidence Score: **{confidence * 100:.1f}%**", unsafe_allow_html=True)

    # All class scores as progress bars
    st.markdown("#### Class Scores")
    for cls_name, score in pred["probabilities"].items():
        pretty = cls_name.replace("Potato___", "").replace("_", " ")
        st.progress(score, text=f"{pretty}:  {score * 100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Disease guidance --------------------------------------------------
    st.markdown("---")
    st.markdown("### 📋 Understanding Your Result")

    info = get_disease_info(pred["predicted_class"])

    st.markdown(f"**{info['display_name']}**")
    st.markdown(info["description"])

    if info["precautions"]:
        st.markdown("**General Guidance:**")
        for item in info["precautions"]:
            st.markdown(f"- {item}")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <b>Disclaimer:</b> This AI model is an educational screening tool "
        "and does not provide a definitive plant disease diagnosis. Predictions "
        "may be incorrect, especially for unclear images or leaves outside the "
        "training data. Always consult an agricultural expert for confirmation."
        "</div>",
        unsafe_allow_html=True,
    )

    # ---- PDF report --------------------------------------------------------
    st.markdown("---")
    try:
        pdf_bytes = generate_report(
            filename=st.session_state.uploaded_name or "image.jpg",
            image_bytes=st.session_state.uploaded_bytes,
            predicted_class=pred["predicted_class"],
            display_name=pred["display_name"],
            confidence=pred["confidence"],
            probabilities=pred["probabilities"],
            disease_description=info["description"],
            precautions=info["precautions"],
        )
        st.download_button(
            label="📄 Download Prediction Report (PDF)",
            data=pdf_bytes,
            file_name="potato_leaf_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except Exception as exc:
        st.warning(f"Could not generate PDF report: {exc}")

    # ---- Reset button ------------------------------------------------------
    st.markdown("")
    if st.button("🔄 Analyze Another Image", use_container_width=True):
        st.session_state.prediction = None
        st.session_state.uploaded_bytes = None
        st.session_state.uploaded_name = None
        st.session_state.analysis_done = False
        st.session_state.uploader_key += 1
        st.rerun()
