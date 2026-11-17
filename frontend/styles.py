"""
styles.py — Custom CSS for the Streamlit frontend.

Returns a string of CSS that is injected via st.markdown(unsafe_allow_html=True).
Keeps the main app.py free of large style blocks.
"""


def get_custom_css() -> str:
    """Return the CSS string for the application."""
    return """
    <style>
    /* ---- Global ---- */
    .stApp {
        background-color: #FAFAFA;
    }
    /* Hide the default Streamlit hamburger & footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* ---- Header area ---- */
    .main-title {
        text-align: center;
        color: #166534;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0.1rem;
    }
    .sub-title {
        text-align: center;
        color: #6B7280;
        font-size: 1.05rem;
        margin-bottom: 0.4rem;
    }
    .instruction {
        text-align: center;
        color: #9CA3AF;
        font-size: 0.92rem;
        margin-bottom: 1.5rem;
    }

    /* ---- Result card ---- */
    .result-card {
        background: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1rem 0;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }

    /* ---- Colour badges ---- */
    .badge-green  { color: #166534; background: #D1FAE5; padding: 4px 14px; border-radius: 20px; font-weight: 600; display: inline-block; }
    .badge-amber  { color: #92400E; background: #FEF3C7; padding: 4px 14px; border-radius: 20px; font-weight: 600; display: inline-block; }
    .badge-red    { color: #991B1B; background: #FEE2E2; padding: 4px 14px; border-radius: 20px; font-weight: 600; display: inline-block; }

    /* ---- Disclaimer box ---- */
    .disclaimer {
        background: #F3F4F6;
        border-left: 4px solid #9CA3AF;
        border-radius: 6px;
        padding: 0.8rem 1rem;
        margin-top: 1rem;
        font-size: 0.85rem;
        color: #6B7280;
    }

    /* ---- Button tweaks ---- */
    .stButton > button {
        border-radius: 8px;
    }
    div[data-testid="stFileUploader"] {
        border-radius: 10px;
    }
    </style>
    """
