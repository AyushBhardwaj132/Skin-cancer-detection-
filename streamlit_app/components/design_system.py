"""Stitch Design System — 17 Reusable UI Components.

Clean modular component library implementing a unified color palette, typography scale,
spacing grid, shadow system, and border radius system.
"""
from __future__ import annotations
from typing import Callable
import streamlit as st

# =============================================================================
# 1. Navbar Component
# =============================================================================
def render_navbar(
    logo_text: str = "DermaVision AI",
    nav_items: list[tuple[str, str]] | None = None,
    active_key: str = "landing",
    action_button_label: str = "Start Analysis"
) -> str:
    """Renders single sticky top navbar matching Stitch TopNavBar spec 1:1."""
    if nav_items is None:
        nav_items = [
            ("landing", "About"),
            ("project", "How It Works"),
            ("upload", "Analysis"),
            ("results", "Model Performance"),
            ("advantages", "Roadmap"),
        ]

    st.markdown(f"""
    <div style="background-color: #ffffff; border-bottom: 1px solid #e3e8ef; height: 80px; display: flex; align-items: center; justify-content: space-between; padding: 0 32px; margin-bottom: 32px; border-radius: 0 0 12px 12px;">
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 24px; font-weight: 600; color: #003757;">
            {logo_text}
        </div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns([3, 1.5, 2, 1.5, 2.5, 1.5, 2])
    with cols[0]:
        st.markdown(f"<div style='margin-top: -56px; font-family: \"Hanken Grotesk\", sans-serif; font-size: 24px; font-weight: 600; color: #003757;'>{logo_text}</div>", unsafe_allow_html=True)

    selected = active_key
    for i, (page_key, label) in enumerate(nav_items):
        is_active = (page_key == active_key)
        btn_type = "primary" if is_active else "secondary"
        if cols[i + 1].button(label, key=f"ds_nav_{page_key}", use_container_width=True, type=btn_type):
            selected = page_key

    if cols[-1].button(action_button_label, key="ds_nav_action", use_container_width=True, type="primary"):
        selected = "upload"

    st.markdown("<hr style='border: none; border-top: 1px solid #e3e8ef; margin: 12px 0 24px 0;'/>", unsafe_allow_html=True)
    return selected


# =============================================================================
# 2. Primary Button Component
# =============================================================================
def render_primary_button(label: str, key: str, use_container_width: bool = True) -> bool:
    """Renders primary button in #1B4E73 container fill."""
    return st.button(label, key=key, use_container_width=use_container_width, type="primary")


# =============================================================================
# 3. Secondary Button Component
# =============================================================================
def render_secondary_button(label: str, key: str, use_container_width: bool = True) -> bool:
    """Renders secondary button with transparent background and hairline border."""
    return st.button(label, key=key, use_container_width=use_container_width, type="secondary")


# =============================================================================
# 4. Card Component
# =============================================================================
def render_card(content_html: str, border_accent: str = "none"):
    """Renders reusable Stitch surface card container."""
    accent_class = ""
    if border_accent == "teal":
        accent_class = "stitch-card-teal-top"
    elif border_accent == "primary":
        accent_class = "stitch-card-primary-top"

    st.markdown(f"""
    <div class="stitch-card {accent_class}">
        {content_html}
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 5. Metric Card Component
# =============================================================================
def render_metric_card(value: str, label: str, color: str = "#003757"):
    """Renders single metric box container."""
    st.markdown(f"""
    <div class="metric-box">
        <div class="val" style="color: {color};">{value}</div>
        <div class="lbl">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 6. Feature Card Component
# =============================================================================
def render_feature_card(title: str, description: str, icon: str, badge_tag: str):
    """Renders feature card with top-right status tag and monoline icon."""
    st.markdown(f"""
    <div class="stitch-card stitch-card-teal-top" style="height: 100%;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px;">
            <div style="background: #0FA3A315; color: #0FA3A3; width: 40px; height: 40px; border-radius: 8px; display: flex; align-items: center; justify-content: center;">
                <span class="material-symbols-outlined">{icon}</span>
            </div>
            <span class="label-caps" style="background: #e6f4ea; color: #137333; padding: 4px 10px; border-radius: 20px;">✓ {badge_tag}</span>
        </div>
        <h4 style="margin: 0 0 8px 0; font-size: 1.1rem; color: #003757; font-family: 'Hanken Grotesk', sans-serif;">{title}</h4>
        <p style="margin: 0; font-size: 0.88rem; color: #475467; line-height: 1.55;">{description}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 7. Section Container Component
# =============================================================================
def render_section_container(title: str, subtitle: str, content_fn: Callable):
    """Renders structured section container with Hanken Grotesk title."""
    st.markdown(f"""
    <div style="margin-bottom: 24px;">
        <h2 class="font-headline-lg" style="margin: 0; font-size: 32px; color: #003757;">{title}</h2>
        <p class="font-body-lg" style="margin-top: 6px; color: #42474e; font-size: 16px;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)
    content_fn()


# =============================================================================
# 8. Hero Section Component
# =============================================================================
def render_hero_section(
    tag: str,
    title: str,
    description: str,
    primary_btn_label: str,
    secondary_btn_label: str,
    image_url: str,
    id_label: str = "ISIC_0015690",
    confidence_val: str = "CONFIDENCE 94%"
):
    """Renders 55%/45% 2-column flex hero section with medical reticle container."""
    col_left, col_right = st.columns([11, 9])
    with col_left:
        st.markdown(f"""
        <div style="display: flex; flex-direction: column; gap: 16px; padding-top: 16px;">
            <span class="label-caps" style="color: #1b4e73; letter-spacing: 0.1em;">{tag}</span>
            <h1 class="font-display-lg" style="margin: 0; font-size: 48px; line-height: 1.15; color: #1a1c1e;">{title}</h1>
            <p class="font-body-lg" style="margin-top: 8px; color: #42474e; font-size: 18px; line-height: 1.6;">{description}</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        b1, b2 = st.columns(2)
        with b1:
            st.button(primary_btn_label, key="ds_hero_p", use_container_width=True, type="primary")
        with b2:
            st.button(secondary_btn_label, key="ds_hero_s", use_container_width=True, type="secondary")

    with col_right:
        st.markdown(f"""
        <div class="hero-wrapper-card">
            <div class="medical-image-container">
                <img src="{image_url}" style="width: 100%; height: 100%; object-fit: cover; opacity: 0.85;" alt="Medical Scan"/>
                <div class="bracket bracket-tl"></div>
                <div class="bracket bracket-tr"></div>
                <div class="bracket bracket-bl"></div>
                <div class="bracket bracket-br"></div>
                <div class="scan-line"></div>
                <div style="position: absolute; top: 16px; left: 16px;">
                    <span class="font-metric-lg" style="font-size: 12px; color: rgba(255,255,255,0.75); font-family: 'JetBrains Mono', monospace;">{id_label}</span>
                </div>
                <div style="position: absolute; bottom: 20px; right: 20px; background: rgba(255,255,255,0.92); border: 1px solid #c2c7cf; border-radius: 6px; padding: 6px 12px; display: flex; align-items: center; gap: 8px;">
                    <div class="pulsing-dot"></div>
                    <span class="font-metric-lg" style="font-size: 13px; color: #1a1c1e; font-family: 'JetBrains Mono', monospace;">{confidence_val}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 9. Timeline Component
# =============================================================================
def render_timeline(milestones: list[dict]):
    """Renders milestone timeline cards."""
    for m in milestones:
        st.markdown(f"""
        <div class="stitch-card" style="margin-bottom: 16px;">
            <div style="display: flex; gap: 16px; align-items: flex-start;">
                <div style="background: #003757; color: white; width: 44px; height: 44px; border-radius: 10px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
                    <span class="material-symbols-outlined">{m.get('icon', 'event')}</span>
                </div>
                <div>
                    <div class="label-caps" style="color: #0FA3A3;">{m.get('step', 'Step')}</div>
                    <h4 style="margin: 2px 0 6px 0; color: #003757; font-size: 1.1rem; font-family: 'Hanken Grotesk', sans-serif;">{m.get('title', '')}</h4>
                    <p style="margin: 0; font-size: 0.88rem; color: #475467; line-height: 1.5;">{m.get('detail', '')}</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 10. Badge Component
# =============================================================================
def render_badge(label: str, status_type: str = "low"):
    """Renders pill status badge."""
    color_map = {
        "low": ("#e6f4ea", "#137333"),
        "mod": ("#fef7e0", "#b06000"),
        "high": ("#fce8e6", "#c5221f"),
    }
    bg, fg = color_map.get(status_type, ("#e6f4ea", "#137333"))
    st.markdown(f"""
    <span class="label-caps" style="background: {bg}; color: {fg}; padding: 6px 16px; border-radius: 20px;">
        ● {label}
    </span>
    """, unsafe_allow_html=True)


# =============================================================================
# 11. Icon Button Component
# =============================================================================
def render_icon_button(icon_name: str, key: str) -> bool:
    """Renders icon button."""
    return st.button(f"<{icon_name}>", key=key)


# =============================================================================
# 12. Footer Component
# =============================================================================
def render_footer(
    brand_name: str = "DermaVision AI",
    disclaimer_text: str = "© 2024 DermaVision AI. All rights reserved.",
    links: list[str] | None = None
):
    """Renders dark ink navy #101828 full-width footer component."""
    if links is None:
        links = ["GitHub", "Research", "Documentation", "Legal", "Contact"]

    links_html = "".join([f'<span class="label-caps" style="color: #94a3b8; cursor: pointer;">{l}</span>' for l in links])

    st.markdown(f"""
    <footer style="background-color: #101828; color: #f0f0f4; padding: 48px 32px; margin-top: 64px; border-radius: 12px 12px 0 0;">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 24px; max-width: 1280px; margin: 0 auto;">
            <div style="max-width: 500px;">
                <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                    {brand_name}
                </div>
                <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.5; color: #cbd5e1; margin: 0;">
                    {disclaimer_text}
                </p>
            </div>
            <div style="display: flex; gap: 24px; align-items: center;">
                {links_html}
            </div>
        </div>
    </footer>
    """, unsafe_allow_html=True)


# =============================================================================
# 13. Input Component
# =============================================================================
def render_custom_input(label: str, key: str, default_val: float = 0.0) -> float:
    """Renders numeric input control."""
    return st.number_input(label, value=default_val, key=key)


# =============================================================================
# 14. Upload Area Component
# =============================================================================
def render_upload_area(title: str = "Dropzone", subtitle: str = "Supported: JPG, PNG"):
    """Renders dashed dropzone upload area."""
    st.markdown(f"""
    <div class="stitch-card" style="border: 2px dashed #0d8abc; background: rgba(13,138,188,0.03); text-align: center; padding: 32px 16px;">
        <span class="material-symbols-outlined" style="font-size: 40px; color: #0d8abc;">cloud_upload</span>
        <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: #003757; margin-top: 8px;">
            {title}
        </div>
        <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
            {subtitle}
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 15. Progress Bar Component
# =============================================================================
def render_progress_bar(percentage: float, color: str = "#0FA3A3", label: str = "Progress"):
    """Renders linear progress bar."""
    st.markdown(f"""
    <div style="margin: 16px 0;">
        <div style="display: flex; justify-content: space-between; font-size: 12px; font-weight: 600; color: #64748b; margin-bottom: 6px;">
            <span>{label}</span>
            <span>{percentage:.1f}%</span>
        </div>
        <div style="background: #e2e8f0; height: 10px; border-radius: 5px; overflow: hidden;">
            <div style="width: {percentage}%; background: {color}; height: 100%; border-radius: 5px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 16. Risk Gauge Component
# =============================================================================
def render_risk_gauge_component(risk_score: float, risk_level: str, risk_color: str):
    """Renders diagnostic prediction risk gauge card."""
    prob_pct = round(risk_score * 100.0, 1)
    st.markdown(f"""
    <div class="stitch-card" style="border-top: 4px solid {risk_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <span class="label-caps" style="color: #64748b;">Diagnostic Risk Score</span>
                <h3 style="margin: 4px 0 0 0; color: #003757; font-family: 'Hanken Grotesk', sans-serif;">{risk_level}</h3>
            </div>
            <span class="label-caps" style="background: {risk_color}15; color: {risk_color}; padding: 6px 16px; border-radius: 20px;">
                ● {risk_level}
            </span>
        </div>
        <div style="background: #e2e8f0; height: 14px; border-radius: 7px; overflow: hidden; position: relative; margin-top: 12px;">
            <div style="width: {prob_pct}%; background: {risk_color}; height: 100%;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# 17. Loading Spinner Component
# =============================================================================
def render_loading_spinner(message: str = "Processing..."):
    """Renders loading spinner state."""
    st.markdown(f"""
    <div style="display: flex; align-items: center; gap: 12px; padding: 16px; background: #f3f3f7; border-radius: 8px;">
        <div class="pulsing-dot"></div>
        <span style="font-family: 'Inter', sans-serif; font-size: 14px; color: #003757; font-weight: 500;">{message}</span>
    </div>
    """, unsafe_allow_html=True)
