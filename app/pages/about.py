import streamlit as st
from app.components.styles import section_header, feature_card


def render():
    st.header("About MediSense AI")

    st.markdown(
        """
        <div class="info-box">
            MediSense AI is an intelligent health risk prediction system designed to address
            the growing challenge of delayed disease diagnosis. It uses advanced machine learning
            techniques to support early detection and assist healthcare professionals in making
            informed clinical decisions.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Methodology
    st.markdown(section_header("Methodology"), unsafe_allow_html=True)

    st.markdown(
        """
        The system uses a **Stacking Ensemble Learning** approach that combines multiple
        machine learning models for superior prediction accuracy:

        **Layer 1 — Base Learners (6 models):**
        - Random Forest (RF)
        - XGBoost (XGB)
        - LightGBM (LGBM)
        - Support Vector Machine (SVM)
        - K-Nearest Neighbors (KNN)
        - Deep Learning MLP (Keras)

        **Layer 2 — Meta-Learner:**
        - Logistic Regression trained on out-of-fold predictions

        **Training Strategy:**
        - 5-fold stratified cross-validation for out-of-fold predictions
        - SMOTE oversampling on training set for class imbalance
        - SHAP explainability for transparent predictions
        """
    )

    # Tech stack
    st.markdown(section_header("Tech Stack"), unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            feature_card(
                "🧠",
                "Machine Learning",
                "scikit-learn, XGBoost, LightGBM, TensorFlow/Keras, SHAP, imbalanced-learn",
            ),
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            feature_card(
                "📊",
                "Visualization",
                "Plotly, Streamlit, Matplotlib, Seaborn",
            ),
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            feature_card(
                "🚀",
                "Deployment",
                "FastAPI, Docker, GitHub Actions, Render",
            ),
            unsafe_allow_html=True,
        )

    # Datasets
    st.markdown(section_header("Datasets"), unsafe_allow_html=True)
    st.markdown(
        """
        | Dataset | Source | Records | Features | License |
        |---------|--------|---------|----------|---------|
        | Heart Disease (Cleveland) | UCI ML Repository | 303 | 13 | CC BY 4.0 |
        | PIMA Indians Diabetes | Kaggle / UCI | 768 | 8 | Public Domain |
        | Indian Liver Patient | UCI ML Repository | 583 | 10 | CC BY 4.0 |
        """
    )

    # Model results
    st.markdown(section_header("Model Results"), unsafe_allow_html=True)
    st.markdown(
        """
        | Dataset | ROC-AUC | F1 Score | Accuracy |
        |---------|---------|----------|----------|
        | Heart Disease | 0.9390 | 0.8372 | 84.78% |
        | Diabetes | 0.8367 | 0.5789 | 72.41% |
        | Liver Disease | 0.7365 | 0.8060 | 70.45% |
        """
    )

    # Team
    st.markdown(section_header("Team"), unsafe_allow_html=True)
    st.markdown(
        """
        - **Developer:** Raval Manav
        - **Supervisor:** Prof. Darshana Patel
        - **Organization:** FICE Education Pvt. Ltd
        - **Institution:** Ahmedabad Institute of Technology
        """
    )

    # Footer
    st.markdown(
        """
        <div class="app-footer">
            MediSense AI v1.0.0 | Built with Python, scikit-learn, and Streamlit<br>
            &copy; 2026 Raval Manav. All rights reserved.
        </div>
        """,
        unsafe_allow_html=True,
    )
