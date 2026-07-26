"""Upload Page view connecting image uploader, lesion crop preview, and metadata form."""
import streamlit as st
from streamlit_app.components.upload_box import render_upload_and_inputs

def render_upload_page():
    """Renders the Upload & Analysis Workspace page."""
    image, metadata, trigger_analysis = render_upload_and_inputs()

    if trigger_analysis:
        if image is None:
            st.error("⚠️ Please select or upload a valid skin lesion image before initiating analysis.")
        else:
            st.session_state["uploaded_image"] = image
            st.session_state["metadata_payload"] = metadata
            st.session_state["current_page"] = "prediction"
            st.rerun()
