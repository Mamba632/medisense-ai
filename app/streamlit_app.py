import streamlit as st

st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("MediSense AI")
st.subheader("Intelligent Health Risk Prediction System")

st.markdown(
    """
    Welcome to **MediSense AI** — an AI-powered platform for early disease risk
    prediction. Navigate using the sidebar to explore:

    - **EDA Explorer** — Interactive exploratory data analysis
    - **Risk Predictor** — Enter patient data and get risk assessment
    - **Model Performance** — Compare model metrics and feature importance
    - **Patient Insights** — Population-level health analytics
    """
)

st.markdown("---")
st.caption("Built by Raval Manav | Supervised by Prof. Darshana Patel")
