import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from app.components.styles import inject_css
from app.components.sidebar import render_sidebar
from app.pages import (
    home,
    eda_explorer,
    risk_predictor,
    model_performance,
    patient_insights,
    about,
)

st.set_page_config(
    page_title="MediSense AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

selected_page, dataset_name = render_sidebar()

if selected_page == "Home":
    home.render()
elif selected_page == "EDA Explorer":
    eda_explorer.render(dataset_name)
elif selected_page == "Risk Predictor":
    risk_predictor.render(dataset_name)
elif selected_page == "Model Performance":
    model_performance.render(dataset_name)
elif selected_page == "Patient Insights":
    patient_insights.render(dataset_name)
elif selected_page == "About":
    about.render()
