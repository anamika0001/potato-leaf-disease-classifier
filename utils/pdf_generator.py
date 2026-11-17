"""
pdf_generator.py — Creates a downloadable PDF prediction report using ReportLab.

The report is generated entirely in memory (BytesIO) so nothing is saved on disk.
"""

import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def _color_for_class(display_name: str) -> colors.Color:
    """Return a ReportLab colour matching the disease category."""
    name_lower = display_name.lower()
    if "early" in name_lower:
        return colors.HexColor("#F59E0B")   # amber
    elif "late" in name_lower:
        return colors.HexColor("#EF4444")   # red
    else:
        return colors.HexColor("#10B981")   # green


def generate_report(
    filename: str,
    image_bytes: bytes,
    predicted_class: str,
    display_name: str,
    confidence: float,
    probabilities: dict[str, float],
    disease_description: str,
    precautions: list[str],
) -> bytes:
    """
    Build a professional one-page PDF and return it as raw bytes.

    Parameters
    ----------
    filename          : original uploaded filename
    image_bytes       : raw bytes of the uploaded leaf image (for thumbnail)
    predicted_class   : e.g. "Potato___Early_blight"
    display_name      : e.g. "Early Blight"
    confidence        : softmax score 0-1
    probabilities     : dict of all three class scores
    disease_description : text paragraph
    precautions       : list of guidance strings

    Returns
    -------
    bytes — the PDF file content
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
        leftMargin=2 * cm,
        rightMargin=2 * cm,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        fontSize=20,
        textColor=colors.HexColor("#166534"),
        spaceAfter=4 * mm,
    )
    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.gray,
        spaceAfter=6 * mm,
    )
    heading_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#166534"),
        spaceBefore=5 * mm,
        spaceAfter=3 * mm,
    )
    body_style = ParagraphStyle(
        "BodyText2",
        parent=styles["BodyText"],
        fontSize=10,
        leading=14,
    )
    small_style = ParagraphStyle(
        "SmallText",
        parent=styles["Normal"],
        fontSize=8,
        textColor=colors.gray,
        spaceBefore=8 * mm,
    )

    story: list = []

    # --- Title ---------------------------------------------------------------
    story.append(Paragraph("🥔 Potato Leaf Disease Detector", title_style))
    story.append(Paragraph("AI-Powered Prediction Report", subtitle_style))

    # --- Meta info -----------------------------------------------------------
    now = datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    meta_data = [
        ["Date & Time:", now],
        ["Uploaded File:", filename or "N/A"],
    ]
    meta_table = Table(meta_data, colWidths=[4 * cm, 12 * cm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ("TOPPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 4 * mm))

    # --- Thumbnail -----------------------------------------------------------
    try:
        img_stream = io.BytesIO(image_bytes)
        rl_img = RLImage(img_stream, width=4 * cm, height=4 * cm, kind="proportional")
        story.append(rl_img)
        story.append(Spacer(1, 4 * mm))
    except Exception:
        story.append(Paragraph("<i>Image thumbnail could not be rendered.</i>", body_style))

    # --- Prediction result ---------------------------------------------------
    story.append(Paragraph("Prediction Result", heading_style))
    accent = _color_for_class(display_name)
    result_data = [
        ["Predicted Class:", display_name],
        ["Confidence Score:", f"{confidence * 100:.1f}%"],
    ]
    result_table = Table(result_data, colWidths=[4 * cm, 12 * cm])
    result_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("TEXTCOLOR", (1, 0), (1, 0), accent),
        ("FONTNAME", (1, 0), (1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(result_table)
    story.append(Spacer(1, 3 * mm))

    # --- All class scores ----------------------------------------------------
    story.append(Paragraph("Class Scores", heading_style))
    score_rows = [["Class", "Score"]]
    for cls_name, score in probabilities.items():
        pretty_name = cls_name.replace("Potato___", "").replace("_", " ")
        score_rows.append([pretty_name, f"{score * 100:.2f}%"])

    score_table = Table(score_rows, colWidths=[8 * cm, 4 * cm])
    score_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#166534")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
    ]))
    story.append(score_table)
    story.append(Spacer(1, 4 * mm))

    # --- Disease description -------------------------------------------------
    story.append(Paragraph("About This Condition", heading_style))
    story.append(Paragraph(disease_description, body_style))
    story.append(Spacer(1, 3 * mm))

    # --- Precautions ---------------------------------------------------------
    if precautions:
        story.append(Paragraph("Recommended Precautions", heading_style))
        for idx, item in enumerate(precautions, start=1):
            story.append(Paragraph(f"{idx}. {item}", body_style))
        story.append(Spacer(1, 3 * mm))

    # --- Disclaimer ----------------------------------------------------------
    story.append(Paragraph(
        "<b>Disclaimer:</b> This AI model is an educational screening tool and "
        "does not provide a definitive plant disease diagnosis. Predictions may "
        "be incorrect, especially for unclear images or leaves outside the training "
        "data. Always consult an agricultural expert for confirmation.",
        small_style,
    ))

    # Build PDF
    doc.build(story)
    return buffer.getvalue()
