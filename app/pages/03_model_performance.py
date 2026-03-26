import streamlit as st
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.metrics import roc_curve, roc_auc_score
from src.config import MODELS_DIR
from src.data.loader import load_clean_dataset, get_target_column
from src.data.splitter import stratified_split
from app.components.charts import roc_curve_plot
from app.components.sidebar import dataset_selector, show_dataset_info

st.header("Model Performance")

dataset_name = dataset_selector()
show_dataset_info(dataset_name)

model_path = MODELS_DIR / dataset_name / "stacking_ensemble.pkl"

if not model_path.exists():
    st.warning(f"No trained model found for {dataset_name}. Run training first.")
    st.stop()

ensemble = joblib.load(model_path)
preprocessor = joblib.load(MODELS_DIR / dataset_name / "preprocessor.pkl")

# Load and prepare test data
df = load_clean_dataset(dataset_name)
target_col = get_target_column(dataset_name)

_, _, test_df = stratified_split(df, target_col)
X_test = preprocessor.transform(test_df.drop(columns=[target_col]))
y_test = test_df[target_col].values

# ROC curves for each base learner + ensemble
fpr_dict, tpr_dict, auc_dict = {}, {}, {}

for name, learner in ensemble.fitted_base_learners_:
    y_prob = learner.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr_dict[name] = fpr
    tpr_dict[name] = tpr
    auc_dict[name] = roc_auc_score(y_test, y_prob)

# Ensemble
y_prob_ens = ensemble.predict_proba(X_test)[:, 1]
fpr_ens, tpr_ens, _ = roc_curve(y_test, y_prob_ens)
fpr_dict["Stacking Ensemble"] = fpr_ens
tpr_dict["Stacking Ensemble"] = tpr_ens
auc_dict["Stacking Ensemble"] = roc_auc_score(y_test, y_prob_ens)

fig = roc_curve_plot(fpr_dict, tpr_dict, auc_dict)
st.plotly_chart(fig, use_container_width=True)

# Model comparison table
st.subheader("Model Comparison")
comparison_data = []
for name in auc_dict:
    comparison_data.append({"Model": name, "ROC-AUC": f"{auc_dict[name]:.4f}"})

st.table(comparison_data)
