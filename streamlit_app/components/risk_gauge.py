"""Risk gauge meter and diagnostic probability summary card matching Stitch design 1:1."""
import streamlit as st

def render_risk_gauge(prediction_results: dict):
    """Renders professional risk gauge meter and metric cards matching Stitch AI Results Dashboard."""
    prob = prediction_results["probability"]
    prob_pct = round(prob * 100.0, 1)
    risk_level = prediction_results["risk_level"]
    risk_color = prediction_results["risk_color"]
    prediction = prediction_results["prediction"]
    confidence_pct = prediction_results["confidence_pct"]
    inference_time = prediction_results["inference_time_ms"]

    st.markdown(f"""
    <div class="stitch-card" style="border-top: 4px solid {risk_color};">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
            <div>
                <span class="label-caps" style="color: #64748b;">Diagnostic AI Prediction Result</span>
                <h2 style="margin: 4px 0 0 0; color: #003757; font-family: 'Hanken Grotesk', sans-serif; font-size: 28px;">{prediction}</h2>
            </div>
            <div style="background: {risk_color}15; color: {risk_color}; padding: 6px 18px; border-radius: 20px; font-weight: 700; font-size: 14px; border: 1px solid {risk_color}30; font-family: 'IBM Plex Sans', sans-serif; text-transform: uppercase; letter-spacing: 0.05em;">
                ● {risk_level}
            </div>
        </div>

        <!-- Linear Risk Meter Progress Bar -->
        <div style="margin: 20px 0;">
            <div style="display: flex; justify-content: space-between; font-size: 12px; font-family: 'IBM Plex Sans', sans-serif; font-weight: 600; color: #64748b; margin-bottom: 6px;">
                <span>Benign (0%)</span>
                <span>Threshold (35%)</span>
                <span>High Risk (65%)</span>
                <span>Malignant (100%)</span>
            </div>
            <div style="background: #e2e8f0; height: 14px; border-radius: 7px; overflow: hidden; position: relative;">
                <div style="background: linear-gradient(90deg, #10b981 0%, #f59e0b 50%, #dc2626 100%); width: 100%; height: 100%; opacity: 0.25;"></div>
                <div style="position: absolute; top: 0; left: 0; bottom: 0; width: {prob_pct}%; background: {risk_color}; border-radius: 7px; transition: width 0.8s ease;"></div>
            </div>
            <div style="text-align: right; font-size: 14px; font-family: 'JetBrains Mono', monospace; font-weight: 600; color: {risk_color}; margin-top: 6px;">
                CALCULATED RISK SCORE: {prob_pct}%
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Metric Readout Boxes matching Stitch spec
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val" style="color: {risk_color};">{prob_pct}%</div>
            <div class="lbl">Probability Score</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val" style="color: #003757;">{confidence_pct}%</div>
            <div class="lbl">Confidence Index</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val" style="color: #0FA3A3;">{inference_time} ms</div>
            <div class="lbl">Inference Latency</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="val" style="font-size: 18px; color: #003757;">EfficientNetV2-S</div>
            <div class="lbl">Active Model Backbone</div>
        </div>
        """, unsafe_allow_html=True)
