import streamlit as st
from src.config import DATASETS

PAGES = [
    "Home",
    "EDA Explorer",
    "Risk Predictor",
    "Model Performance",
    "Patient Insights",
    "About",
]

PAGES_NEEDING_DATASET = {
    "EDA Explorer",
    "Risk Predictor",
    "Model Performance",
    "Patient Insights",
}


def render_sidebar():
    st.sidebar.markdown(
        """
        <div style="text-align:center; padding: 1rem 0;">
            <h2 style="color:#0077B6; margin-bottom:0;">MediSense AI</h2>
            <p style="color:#6C757D; font-size:0.85rem;">Intelligent Health Risk Prediction</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")

    selected_page = st.sidebar.radio("Navigation", PAGES, label_visibility="collapsed")

    dataset_name = None
    if selected_page in PAGES_NEEDING_DATASET:
        st.sidebar.markdown("---")
        dataset_name = st.sidebar.selectbox(
            "Select Disease Dataset",
            options=list(DATASETS.keys()),
            format_func=lambda x: DATASETS[x]["description"],
        )
        config = DATASETS[dataset_name]
        st.sidebar.caption(f"Target: `{config['target']}`")
        st.sidebar.caption(
            f"Features: {len(config['numeric_features'])} numeric, "
            f"{len(config['categorical_features'])} categorical"
        )

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 | Raval Manav")

    return selected_page, dataset_name
