"""Pixel-perfect upload card & metadata workspace matching Stitch design 1:1."""
from __future__ import annotations
import numpy as np
from PIL import Image
import streamlit as st

from streamlit_app.config.settings import CATEGORICAL_FIELDS, DEFAULT_METADATA_VALUES
from streamlit_app.utils.image_utils import validate_image_file, crop_lesion_centered


def render_upload_and_inputs() -> tuple[Image.Image | None, dict | None, bool]:
    """Renders upload workspace matching Stitch 2-column layout."""
    st.markdown("""
    <div style="margin-bottom: 24px;">
        <h1 class="font-headline-lg" style="margin: 0; font-size: 32px; color: #003757;">
            Skin Lesion Upload & Metadata Analysis
        </h1>
        <p class="font-body-lg" style="margin-top: 6px; color: #42474e; font-size: 16px;">
            Upload a dermoscopic or clinical image and complete patient metadata for multimodal diagnostic evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_form = st.columns([1, 1])
    uploaded_image = None
    metadata_inputs = None
    trigger_analysis = False

    with col_img:
        st.markdown("""
        <div class="stitch-card" style="border: 2px dashed #0d8abc; background: rgba(13,138,188,0.03); text-align: center; padding: 32px 16px;">
            <span class="material-symbols-outlined" style="font-size: 40px; color: #0d8abc;">cloud_upload</span>
            <div style="font-family: 'Hanken Grotesk', sans-serif; font-size: 18px; font-weight: 600; color: #003757; margin-top: 8px;">
                Dermoscopic Lesion Image Dropzone
            </div>
            <div style="font-size: 13px; color: #64748b; margin-top: 4px;">
                Supported Formats: JPG, JPEG, PNG (Max 20MB)
            </div>
        </div>
        """, unsafe_allow_html=True)

        file_buffer = st.file_uploader(
            "Select Dermoscopic Image",
            type=["jpg", "jpeg", "png"],
            key="lesion_image_uploader",
            label_visibility="collapsed"
        )

        if file_buffer is not None:
            is_valid, msg, image = validate_image_file(file_buffer)
            if is_valid and image is not None:
                uploaded_image = image
                st.success(f"✓ Image loaded successfully ({image.size[0]}x{image.size[1]} px)")

                # Generate Lesion Center Crop Preview
                img_np = np.array(image.convert("RGB"))
                cropped_np = crop_lesion_centered(img_np)
                cropped_img = Image.fromarray(cropped_np)

                prev_c1, prev_c2 = st.columns(2)
                with prev_c1:
                    st.markdown("<span class='label-caps' style='color:#003757;'>Original Image</span>", unsafe_allow_html=True)
                    st.image(image, use_container_width=True)
                with prev_c2:
                    st.markdown("<span class='label-caps' style='color:#0FA3A3;'>Lesion Center Crop (ROI)</span>", unsafe_allow_html=True)
                    st.image(cropped_img, use_container_width=True)

            else:
                st.error(msg)

    with col_form:
        st.markdown("""
        <div class="stitch-card">
            <span class="label-caps" style="color: #1b4e73; display: block; margin-bottom: 12px;">Patient Demographics & 3D Metadata</span>
        </div>
        """, unsafe_allow_html=True)

        with st.form(key="metadata_submission_form"):
            c_meta1, c_meta2 = st.columns(2)
            with c_meta1:
                age = st.number_input("Patient Approx Age", min_value=1.0, max_value=100.0, value=DEFAULT_METADATA_VALUES["age_approx"], step=1.0)
                sex = st.selectbox("Sex", CATEGORICAL_FIELDS["sex"], index=0)
                anatom_site = st.selectbox("Anatomical Site", CATEGORICAL_FIELDS["anatom_site_general"], index=2)
                image_type = st.selectbox("Image Type", CATEGORICAL_FIELDS["image_type"], index=0)

            with c_meta2:
                tile_type = st.selectbox("3D TBP Tile Type", CATEGORICAL_FIELDS["tbp_tile_type"], index=0)
                location = st.selectbox("3D Lesion Location", CATEGORICAL_FIELDS["tbp_lv_location"], index=4)
                clin_size = st.number_input("Clin Size Diam (mm)", min_value=0.1, max_value=50.0, value=DEFAULT_METADATA_VALUES["clin_size_long_diam_mm"], step=0.5)
                nevi_conf = st.slider("Nevi Confidence", min_value=0.0, max_value=1.0, value=DEFAULT_METADATA_VALUES["tbp_lv_nevi_confidence"], step=0.05)

            with st.expander("🔬 Advanced 3D Spatial & Color Metrics", expanded=False):
                ca1, ca2 = st.columns(2)
                with ca1:
                    area_mm2 = st.number_input("Area (mm²)", value=DEFAULT_METADATA_VALUES["tbp_lv_areaMM2"])
                    perim_mm = st.number_input("Perimeter (mm)", value=DEFAULT_METADATA_VALUES["tbp_lv_perimeterMM"])
                    color_std = st.number_input("Color Std Mean", value=DEFAULT_METADATA_VALUES["tbp_lv_color_std_mean"])
                with ca2:
                    eccentricity = st.number_input("Eccentricity", value=DEFAULT_METADATA_VALUES["tbp_lv_eccentricity"])
                    norm_border = st.number_input("Norm Border", value=DEFAULT_METADATA_VALUES["tbp_lv_norm_border"])
                    norm_color = st.number_input("Norm Color", value=DEFAULT_METADATA_VALUES["tbp_lv_norm_color"])

            submit_btn = st.form_submit_button(
                "Run AI Diagnostic Analysis",
                use_container_width=True,
                type="primary"
            )

            if submit_btn:
                trigger_analysis = True
                metadata_inputs = dict(DEFAULT_METADATA_VALUES)
                metadata_inputs.update({
                    "age_approx": float(age),
                    "sex": sex,
                    "anatom_site_general": anatom_site,
                    "image_type": image_type,
                    "tbp_tile_type": tile_type,
                    "tbp_lv_location": location,
                    "tbp_lv_location_simple": location.split()[0] if " " in location else location,
                    "clin_size_long_diam_mm": float(clin_size),
                    "tbp_lv_nevi_confidence": float(nevi_conf),
                    "tbp_lv_areaMM2": float(area_mm2),
                    "tbp_lv_perimeterMM": float(perim_mm),
                    "tbp_lv_color_std_mean": float(color_std),
                    "tbp_lv_eccentricity": float(eccentricity),
                    "tbp_lv_norm_border": float(norm_border),
                    "tbp_lv_norm_color": float(norm_color),
                })

    return uploaded_image, metadata_inputs, trigger_analysis
