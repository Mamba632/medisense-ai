import shap
import matplotlib.pyplot as plt
from pathlib import Path


def shap_summary(model, X, feature_names=None, save_path=None):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X, 50))

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values


def shap_waterfall_single(model, X_single, feature_names=None, save_path=None):
    try:
        explainer = shap.TreeExplainer(model)
    except Exception:
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_single, 50))

    shap_values = explainer(X_single)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        plt.figure(figsize=(10, 6))
        shap.waterfall_plot(shap_values[0], show=False)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

    return shap_values
