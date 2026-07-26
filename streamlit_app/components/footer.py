"""Pixel-perfect Footer component matching Stitch design system."""
import streamlit as st

def render_footer():
    """Renders bottom medical disclaimer footer matching Stitch design 1:1."""
    st.markdown("""
    <footer style="background-color: #101828; color: #f0f0f4; padding: 48px 32px; margin-top: 64px; border-radius: 12px 12px 0 0;">
        <div style="display: flex; flex-wrap: wrap; justify-content: space-between; align-items: flex-start; gap: 24px; max-width: 1280px; margin: 0 auto;">
            <div style="max-width: 500px;">
                <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 24px; font-weight: 700; color: #ffffff; margin-bottom: 12px;">
                    DermaVision AI
                </div>
                <p style="font-family: 'Inter', sans-serif; font-size: 14px; line-height: 1.5; color: #cbd5e1; margin: 0;">
                    © 2024 DermaVision AI. All rights reserved. Medical Disclaimer: This AI tool is for research assistance only and does not substitute professional medical advice, diagnosis, or treatment.
                </p>
            </div>
            <div style="display: flex; gap: 24px; align-items: center;">
                <span class="label-caps" style="color: #94a3b8; cursor: pointer;">GitHub</span>
                <span class="label-caps" style="color: #94a3b8; cursor: pointer;">Research</span>
                <span class="label-caps" style="color: #94a3b8; cursor: pointer;">Documentation</span>
                <span class="label-caps" style="color: #94a3b8; cursor: pointer;">Legal</span>
                <span class="label-caps" style="color: #94a3b8; cursor: pointer;">Contact</span>
            </div>
        </div>
    </footer>
    """, unsafe_allow_html=True)
