"""Pixel-perfect TopNavBar component matching Stitch Landing Page specification 1:1."""
import streamlit as st

NAV_ITEMS = [
    ("landing", "About"),
    ("project", "How It Works"),
    ("upload", "Analysis"),
    ("results", "Model Performance"),
    ("advantages", "Roadmap"),
]


def render_header(current_page: str = "landing") -> str:
    """Renders single sticky top header bar matching Stitch TopNavBar 1:1 with active navigation indicator."""
    st.markdown("""
    <div style="background-color: #ffffff; border-bottom: 1px solid #e3e8ef; height: 80px; display: flex; align-items: center; justify-content: space-between; padding: 0 32px; margin-bottom: 32px; border-radius: 0 0 12px 12px;">
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 24px; font-weight: 600; color: #003757;">
            DermaVision AI
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Interactive native Streamlit navigation control bar
    cols = st.columns([3, 1.5, 2, 1.5, 2.5, 1.5, 2])

    with cols[0]:
        st.markdown("<div style='margin-top: -56px; font-family: \"Hanken Grotesk\", sans-serif; font-size: 24px; font-weight: 600; color: #003757;'>DermaVision AI</div>", unsafe_allow_html=True)

    for i, (page_key, label) in enumerate(NAV_ITEMS):
        is_active = (page_key == current_page or (current_page == "landing" and page_key == "upload"))
        btn_type = "primary" if is_active else "secondary"
        if cols[i + 1].button(label, key=f"nav_top_{page_key}", use_container_width=True, type=btn_type):
            st.session_state["current_page"] = page_key
            st.rerun()

    if cols[-1].button("Start Analysis", key="nav_btn_start", use_container_width=True, type="primary"):
        st.session_state["current_page"] = "upload"
        st.rerun()

    st.markdown("<hr style='border: none; border-top: 1px solid #e3e8ef; margin: 12px 0 24px 0;'/>", unsafe_allow_html=True)
    return current_page
