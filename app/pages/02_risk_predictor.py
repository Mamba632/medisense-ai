import streamlit as st
import requests
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import FASTAPI_URL, DATASETS
from app.components.charts import risk_gauge
from app.components.sidebar import dataset_selector

st.header("Risk Predictor")

dataset_name = dataset_selector()
config = DATASETS[dataset_name]

st.subheader(f"Enter Patient Data — {config['description']}")

input_data = {}
cols = st.columns(2)

all_features = config["numeric_features"] + config["categorical_features"]
for i, feature in enumerate(all_features):
    with cols[i % 2]:
        if feature == "Gender":
            input_data[feature] = st.selectbox(feature, ["Male", "Female"])
        elif feature in config["categorical_features"]:
            input_data[feature] = st.number_input(feature, value=0.0, step=1.0)
        else:
            input_data[feature] = st.number_input(feature, value=0.0)

if st.button("Predict Risk", type="primary"):
    try:
        response = requests.post(
            f"{FASTAPI_URL}/predict/{dataset_name}",
            json=input_data,
            timeout=30,
        )
        if response.status_code == 200:
            result = response.json()

            col1, col2 = st.columns(2)
            with col1:
                fig = risk_gauge(result["risk_probability"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                st.metric("Risk Level", result["risk_label"])
                st.metric("Probability", f"{result['risk_probability']:.2%}")
                st.metric("Model Version", result["model_version"])
        else:
            st.error(f"API Error: {response.text}")
    except requests.ConnectionError:
        st.error(
            "Cannot connect to API server. "
            "Make sure FastAPI is running on " + FASTAPI_URL
        )
