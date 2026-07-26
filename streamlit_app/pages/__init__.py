"""Pages package initialization."""
from streamlit_app.pages.landing import render_landing_page
from streamlit_app.pages.upload import render_upload_page
from streamlit_app.pages.prediction import render_prediction_page
from streamlit_app.pages.results import render_results_page
from streamlit_app.pages.project import render_project_page
from streamlit_app.pages.advantages import render_advantages_page
from streamlit_app.pages.about import render_about_page

__all__ = [
    "render_landing_page",
    "render_upload_page",
    "render_prediction_page",
    "render_results_page",
    "render_project_page",
    "render_advantages_page",
    "render_about_page",
]
