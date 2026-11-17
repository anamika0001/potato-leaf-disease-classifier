"""
disease_info.py — Disease descriptions and guidance for each predicted class.

This module stores the textual information shown to the user after a prediction.
Each disease entry contains:
  - display_name : human-readable label
  - color        : UI accent colour name (used by the frontend for styling)
  - description  : brief plain-language explanation of the condition
  - precautions  : list of actionable guidance items
"""

# ---------------------------------------------------------------------------
# Disease / class information dictionary
# ---------------------------------------------------------------------------

DISEASE_INFO = {
    "Potato___Early_blight": {
        "display_name": "Early Blight",
        "color": "amber",
        "description": (
            "Early blight is a common fungal disease of potato plants caused by "
            "*Alternaria solani*. It typically appears as brown spots on older "
            "leaves, often with distinctive concentric ring patterns (target-like "
            "spots). The disease usually starts on lower leaves and progresses "
            "upward if conditions remain favourable."
        ),
        "precautions": [
            "Inspect other leaves on the same plant and neighbouring plants for similar symptoms.",
            "Keep plants appropriately spaced to allow good air circulation.",
            "Avoid prolonged unnecessary leaf wetness — water at the base of the plant rather than overhead.",
            "Remove and safely dispose of heavily affected leaves to reduce spread.",
            "Consult a local agricultural expert or extension worker for recommended fungicide treatment.",
        ],
    },
    "Potato___Late_blight": {
        "display_name": "Late Blight",
        "color": "red",
        "description": (
            "Late blight is a serious potato disease caused by the oomycete "
            "*Phytophthora infestans*. It can spread very quickly under cool, "
            "wet conditions and may destroy an entire crop if not managed in time. "
            "Symptoms include dark, water-soaked lesions on leaves and stems, "
            "sometimes accompanied by a white mould-like growth on the underside "
            "of leaves."
        ),
        "precautions": [
            "Inspect surrounding plants immediately for similar symptoms.",
            "Avoid moving potentially infected plant material to other areas.",
            "Do not compost infected plant debris — dispose of it safely.",
            "Seek timely guidance from an agricultural extension worker or plant disease specialist.",
            "Consider protective fungicide applications as recommended by an expert.",
        ],
    },
    "Potato___healthy": {
        "display_name": "Healthy",
        "color": "green",
        "description": (
            "The model identified characteristics consistent with its healthy-leaf "
            "training class. This suggests the uploaded leaf image does not show "
            "obvious signs of Early Blight or Late Blight as represented in the "
            "training data."
        ),
        "precautions": [
            "Continue regular monitoring of your plants.",
            "Maintain appropriate watering and growing conditions.",
            "Check leaves periodically for new symptoms.",
            "Practice crop rotation and good field hygiene to prevent future disease.",
        ],
    },
}

# Quick lookup helper
def get_disease_info(class_name: str) -> dict:
    """Return the information dict for a given class name, or a fallback."""
    return DISEASE_INFO.get(class_name, {
        "display_name": class_name,
        "color": "gray",
        "description": "No additional information is available for this class.",
        "precautions": [],
    })
