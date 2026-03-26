import streamlit as st
import pandas as pd

from app.components.model_loader import (
    load_model,
    load_preprocessor,
    load_data,
    is_model_available,
)
from app.components.charts import risk_gauge, shap_bar_chart
from app.components.styles import risk_badge, metric_card
from src.config import DATASETS

# Human-readable labels for categorical features
HEART_LABELS = {
    "sex": {0: "Female", 1: "Male"},
    "cp": {
        0: "Typical Angina",
        1: "Atypical Angina",
        2: "Non-anginal Pain",
        3: "Asymptomatic",
    },
    "fbs": {0: "No (<=120 mg/dl)", 1: "Yes (>120 mg/dl)"},
    "restecg": {0: "Normal", 1: "ST-T Abnormality", 2: "LV Hypertrophy"},
    "exang": {0: "No", 1: "Yes"},
    "slope": {0: "Upsloping", 1: "Flat", 2: "Downsloping"},
}


def _get_defaults(df, config):
    defaults = {}
    for col in config["numeric_features"]:
        if col in df.columns:
            defaults[col] = float(df[col].median())
    for col in config["categorical_features"]:
        if col in df.columns:
            defaults[col] = df[col].mode().iloc[0] if not df[col].mode().empty else 0
    return defaults


def render(dataset_name):
    st.header("Risk Predictor")

    if not is_model_available(dataset_name):
        st.warning(f"No trained model found for {dataset_name}. Run training first.")
        return

    config = DATASETS[dataset_name]
    df = load_data(dataset_name)
    defaults = _get_defaults(df, config)

    st.markdown(
        f"""
        <div class="info-box">
            Enter patient health indicators below to predict <strong>{config["description"]}</strong> risk.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Sample patient button
    if st.button("Fill with Sample Patient"):
        st.session_state["use_sample"] = True

    use_sample = st.session_state.get("use_sample", False)

    # Input form
    input_data = {}
    all_features = config["numeric_features"] + config["categorical_features"]
    cols = st.columns(2)

    for i, feature in enumerate(all_features):
        with cols[i % 2]:
            default_val = defaults.get(feature, 0)

            if feature == "Gender":
                options = ["Male", "Female"]
                idx = 0 if use_sample else 0
                input_data[feature] = st.selectbox("Gender", options, index=idx)
            elif dataset_name == "heart" and feature in HEART_LABELS:
                label_map = HEART_LABELS[feature]
                options = list(label_map.values())
                default_idx = int(default_val) if use_sample else 0
                default_idx = min(default_idx, len(options) - 1)
                selected = st.selectbox(feature, options, index=default_idx)
                input_data[feature] = [
                    k for k, v in label_map.items() if v == selected
                ][0]
            elif feature in config["categorical_features"]:
                input_data[feature] = st.number_input(
                    feature,
                    value=float(default_val) if use_sample else 0.0,
                    step=1.0,
                )
            else:
                input_data[feature] = st.number_input(
                    feature,
                    value=float(default_val) if use_sample else 0.0,
                    format="%.2f",
                )

    st.markdown("---")

    if st.button("Predict Risk", type="primary", use_container_width=True):
        ensemble = load_model(dataset_name)
        preprocessor = load_preprocessor(dataset_name)

        input_df = pd.DataFrame([input_data])
        X = preprocessor.transform(input_df)
        prob = float(ensemble.predict_proba(X)[0, 1])

        if prob < 0.3:
            label = "Low"
        elif prob < 0.7:
            label = "Medium"
        else:
            label = "High"

        # Results
        c1, c2 = st.columns([2, 1])
        with c1:
            fig = risk_gauge(prob, f"{config['description']} Risk")
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(risk_badge(label), unsafe_allow_html=True)
            st.markdown(
                metric_card("Risk Probability", f"{prob:.1%}"),
                unsafe_allow_html=True,
            )

        # SHAP explanations
        st.markdown(
            '<div class="section-header">Explanation — Why This Prediction?</div>',
            unsafe_allow_html=True,
        )

        try:
            import shap

            # Use a tree-based base learner for SHAP
            tree_learner = None
            for name, learner in ensemble.fitted_base_learners_:
                if name in ("xgb", "rf", "lgbm"):
                    tree_learner = learner
                    break

            if tree_learner is not None:
                explainer = shap.TreeExplainer(tree_learner)
                shap_values = explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]

                feature_names = preprocessor.get_feature_names_out()
                fig = shap_bar_chart(shap_values[0], feature_names)
                st.plotly_chart(fig, use_container_width=True)

                # Text explanation
                top_features = sorted(
                    zip(feature_names, shap_values[0]),
                    key=lambda x: abs(x[1]),
                    reverse=True,
                )[:5]
                with st.expander("Detailed Feature Contributions"):
                    for fname, sval in top_features:
                        direction = "increases" if sval > 0 else "decreases"
                        st.write(f"- **{fname}** {direction} risk by {abs(sval):.4f}")
        except Exception:
            st.info("SHAP explanations unavailable for this model configuration.")

    if use_sample:
        st.session_state["use_sample"] = False
