"""Technical Architecture & Project Rationale page matching Stitch design."""
import streamlit as st

def render_project_page():
    """Renders Technical Architecture and engineering decision explanations."""
    st.markdown("""
    <div class="stitch-card stitch-card-primary-top">
        <h2 style="margin-top: 0; color: #003757;">Technical Architecture & Engineering Rationale</h2>
        <p style="color: #475467; font-size: 0.95rem;">
            DermaVision AI combines deep convolutional visual representations with multi-tabular metadata MLP features. 
            Below is the comprehensive technical justification for each core architectural choice.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Architecture Diagram (SVG / Flow Render)
    st.markdown("### 🏗️ End-to-End System Pipeline Diagram")
    st.markdown("""
    <div class="stitch-card" style="background: #ffffff; text-align: center; padding: 32px 16px;">
        <svg width="100%" height="220" viewBox="0 0 900 220" fill="none" xmlns="http://www.w3.org/2000/svg">
            <!-- Image Input Box -->
            <rect x="20" y="30" width="140" height="70" rx="10" fill="#F3F3F7" stroke="#003757" stroke-width="2"/>
            <text x="90" y="62" text-anchor="middle" fill="#003757" font-weight="700" font-family="Inter" font-size="13">Skin Image Input</text>
            <text x="90" y="80" text-anchor="middle" fill="#64748B" font-size="11">JPG/PNG (384x384)</text>

            <!-- Lesion Crop Box -->
            <rect x="200" y="30" width="140" height="70" rx="10" fill="#0FA3A310" stroke="#0FA3A3" stroke-width="2"/>
            <text x="270" y="62" text-anchor="middle" fill="#0FA3A3" font-weight="700" font-family="Inter" font-size="13">Lesion Center Crop</text>
            <text x="270" y="80" text-anchor="middle" fill="#64748B" font-size="11">OpenCV Contour ROI</text>

            <!-- Backbone Box -->
            <rect x="380" y="30" width="150" height="70" rx="10" fill="#003757" stroke="#003757" stroke-width="2"/>
            <text x="455" y="62" text-anchor="middle" fill="#FFFFFF" font-weight="700" font-family="Inter" font-size="13">EfficientNetV2-S</text>
            <text x="455" y="80" text-anchor="middle" fill="#7FF5F4" font-size="11">CNN Embeddings (1280-d)</text>

            <!-- Metadata Input Box -->
            <rect x="20" y="130" width="320" height="60" rx="10" fill="#F3F3F7" stroke="#003757" stroke-width="2"/>
            <text x="180" y="157" text-anchor="middle" fill="#003757" font-weight="700" font-family="Inter" font-size="13">47 Tabular Patient & 3D TBP Features</text>
            <text x="180" y="174" text-anchor="middle" fill="#64748B" font-size="11">Age, Site, 3D Spatial & Color Metrics</text>

            <!-- Metadata MLP Box -->
            <rect x="380" y="130" width="150" height="60" rx="10" fill="#0FA3A3" stroke="#0FA3A3" stroke-width="2"/>
            <text x="455" y="157" text-anchor="middle" fill="#FFFFFF" font-weight="700" font-family="Inter" font-size="13">Metadata MLP</text>
            <text x="455" y="174" text-anchor="middle" fill="#E2E8F0" font-size="11">Dense Embeddings (128-d)</text>

            <!-- Fusion Layer Box -->
            <rect x="570" y="70" width="130" height="80" rx="10" fill="#1B4E73" stroke="#1B4E73" stroke-width="2"/>
            <text x="635" y="105" text-anchor="middle" fill="#FFFFFF" font-weight="700" font-family="Inter" font-size="13">Metadata Fusion</text>
            <text x="635" y="125" text-anchor="middle" fill="#CDE5FF" font-size="11">Concat (1408-d)</text>

            <!-- Classifier Output Box -->
            <rect x="740" y="70" width="140" height="80" rx="10" fill="#10B981" stroke="#10B981" stroke-width="2"/>
            <text x="810" y="105" text-anchor="middle" fill="#FFFFFF" font-weight="700" font-family="Inter" font-size="13">Diagnostic Result</text>
            <text x="810" y="125" text-anchor="middle" fill="#E6F4EA" font-size="11">Prob & Risk Level</text>

            <!-- Arrows -->
            <line x1="160" y1="65" x2="200" y2="65" stroke="#003757" stroke-width="2" marker-end="url(#arrow)"/>
            <line x1="340" y1="65" x2="380" y2="65" stroke="#0FA3A3" stroke-width="2"/>
            <line x1="530" y1="65" x2="570" y2="95" stroke="#003757" stroke-width="2"/>
            <line x1="340" y1="160" x2="380" y2="160" stroke="#003757" stroke-width="2"/>
            <line x1="530" y1="160" x2="570" y2="125" stroke="#0FA3A3" stroke-width="2"/>
            <line x1="700" y1="110" x2="740" y2="110" stroke="#10B981" stroke-width="2"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)

    # 6 Decision Justification Cards
    st.markdown("### 🎯 Core Architecture Decisions")

    decisions = [
        {
            "title": "Why EfficientNetV2-S?",
            "icon": "memory",
            "desc": "EfficientNetV2-S utilizes progressive learning and fused Mobile Inverted Bottleneck convolutions (Fused-MBConv). It achieves faster training latency, superior parameter efficiency (21.5M parameters), and higher representation accuracy than ResNet or ViT backbones on high-resolution medical imagery."
        },
        {
            "title": "Why Metadata Fusion?",
            "icon": "hub",
            "desc": "Single-modality image models miss essential patient context. Fusing image representations with 47 tabular metadata features (age, anatom site, 3D color differentials, relative lesion diameter, background skin contrast) improves discriminative capacity for subtle early-stage malignant melanoma."
        },
        {
            "title": "Why GroupKFold Cross-Validation?",
            "icon": "groups",
            "desc": "Individual patients in the ISIC dataset often have multiple skin lesions. Standard random K-Fold splits cause severe data leakage by placing lesions from the same patient in both train and validation sets. GroupKFold groups by `patient_id` to guarantee zero patient overlap."
        },
        {
            "title": "Why Lesion Center Crop?",
            "icon": "crop_free",
            "desc": "Full-field 3D photography images contain large areas of healthy skin, clothing edges, or lighting artifacts. OpenCV Otsu contour bounding box cropping isolates the central lesion ROI with a 20% margin, concentrating model attention on actual cellular structures."
        },
        {
            "title": "Why Focal Loss?",
            "icon": "balance",
            "desc": "Malignant skin cancer cases account for under 1% of total screening samples (extreme positive/negative class imbalance). Focal Loss downweights easy negative background samples using a modulating factor (1 - p)^γ, focusing gradient updates on hard positive malignant lesions."
        },
        {
            "title": "Why ISIC 2024 Dataset?",
            "icon": "dataset",
            "desc": "The ISIC 2024 Challenge dataset represents the gold-standard 3D Whole-Body Photography dataset curated by international dermatological institutes. It includes standardized 3D spatial coordinates, standardized illumination, and ground-truth histopathology labels."
        },
    ]

    cols = st.columns(2)
    for idx, d in enumerate(decisions):
        col = cols[idx % 2]
        with col:
            st.markdown(f"""
            <div class="stitch-card stitch-card-primary-top" style="height: 100%;">
                <div style="display: flex; gap: 12px; align-items: center; margin-bottom: 10px;">
                    <span class="material-symbols-outlined" style="color: #003757;">{d['icon']}</span>
                    <h4 style="margin: 0; color: #003757; font-size: 1.1rem;">{d['title']}</h4>
                </div>
                <p style="margin: 0; font-size: 0.88rem; color: #475467; line-height: 1.55;">{d['desc']}</p>
            </div>
            """, unsafe_allow_html=True)
