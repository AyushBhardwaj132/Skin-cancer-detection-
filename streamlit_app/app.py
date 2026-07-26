"""DermaVision AI — Main Streamlit Application Entry Point."""
import sys
from pathlib import Path
import streamlit as st

# Add project base directory to sys.path to enable imports
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from streamlit_app.components import render_header, render_footer
from streamlit_app.pages import (
    render_landing_page,
    render_upload_page,
    render_prediction_page,
    render_results_page,
    render_project_page,
    render_advantages_page,
    render_about_page,
)


def load_custom_css():
    """Injects custom Stitch CSS styles and typography."""
    css_path = BASE_DIR / "streamlit_app" / "styles" / "stitch_theme.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def main():
    st.set_page_config(
        page_title="DermaVision AI | Clinical Diagnostic Platform",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    load_custom_css()

    # Session state initialization for page routing
    if "current_page" not in st.session_state:
        st.session_state["current_page"] = "landing"

    current_page = st.session_state["current_page"]

    # Render Glassmorphic Navigation Header
    render_header(current_page)

    # Main Page Routing
    if current_page == "landing":
        render_landing_page()
    elif current_page == "upload":
        render_upload_page()
    elif current_page == "prediction":
        render_prediction_page()
    elif current_page == "results":
        render_results_page()
    elif current_page == "project":
        render_project_page()
    elif current_page == "advantages":
        render_advantages_page()
    elif current_page == "about":
        render_about_page()
    else:
        render_landing_page()

    # Render Clinical Disclaimer Footer
    render_footer()


if __name__ == "__main__":
    main()
