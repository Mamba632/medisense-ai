import streamlit as st
from app.components.styles import metric_card, feature_card


def render():
    # Hero banner
    st.markdown(
        """
        <div class="hero-banner">
            <h1>MediSense AI</h1>
            <p>An AI-powered platform for early disease risk prediction using
            advanced machine learning and stacking ensemble techniques.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Key stats
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(metric_card("Diseases Covered", "3"), unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card("ML Models", "6", "success"), unsafe_allow_html=True)
    with c3:
        st.markdown(
            metric_card("Best ROC-AUC", "0.939", "warning"), unsafe_allow_html=True
        )
    with c4:
        st.markdown(metric_card("Patients Analyzed", "1,654"), unsafe_allow_html=True)

    # How it works
    st.markdown(
        '<div class="section-header">How It Works</div>', unsafe_allow_html=True
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            feature_card(
                "📊",
                "Data Analysis",
                "Comprehensive EDA on patient health records to uncover "
                "patterns, correlations, and key health indicators.",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            feature_card(
                "🤖",
                "Risk Prediction",
                "Stacking ensemble of 6 ML models (RF, XGBoost, LightGBM, "
                "SVM, KNN, Deep Learning) for accurate risk assessment.",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            feature_card(
                "🔍",
                "Explainable AI",
                "SHAP-based explanations show which health factors "
                "contribute most to each patient's risk prediction.",
            ),
            unsafe_allow_html=True,
        )

    # Datasets
    st.markdown('<div class="section-header">Datasets</div>', unsafe_allow_html=True)
    st.markdown(
        """
        | Dataset | Records | Features | Target |
        |---------|---------|----------|--------|
        | Heart Disease (Cleveland) | 303 | 13 | Heart disease presence |
        | PIMA Indians Diabetes | 768 | 8 | Diabetes diagnosis |
        | Indian Liver Patient | 583 | 10 | Liver disease status |
        """
    )

    # Team
    st.markdown('<div class="section-header">Team</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            """
            <div class="info-box">
                <strong>Developer:</strong> Raval Manav<br>
                <strong>Supervisor:</strong> Prof. Darshana Patel<br>
                <strong>Institution:</strong> FICE Education Pvt. Ltd
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            """
            <div class="info-box">
                <strong>Project:</strong> MediSense AI<br>
                <strong>Domain:</strong> Healthcare & Medical Analytics<br>
                <strong>Approach:</strong> Stacking Ensemble Learning
            </div>
            """,
            unsafe_allow_html=True,
        )
