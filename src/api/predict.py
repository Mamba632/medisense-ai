import joblib
import pandas as pd

from src.config import MODELS_DIR


_cache = {}


def _load_model(dataset_name: str):
    if dataset_name not in _cache:
        model_dir = MODELS_DIR / dataset_name
        _cache[dataset_name] = {
            "ensemble": joblib.load(model_dir / "stacking_ensemble.pkl"),
            "preprocessor": joblib.load(model_dir / "preprocessor.pkl"),
        }
    return _cache[dataset_name]


def predict_risk(dataset_name: str, input_data: dict) -> dict:
    artifacts = _load_model(dataset_name)
    ensemble = artifacts["ensemble"]
    preprocessor = artifacts["preprocessor"]

    df = pd.DataFrame([input_data])
    X = preprocessor.transform(df)
    prob = ensemble.predict_proba(X)[0, 1]

    if prob < 0.3:
        label = "Low"
    elif prob < 0.7:
        label = "Medium"
    else:
        label = "High"

    return {
        "disease": dataset_name,
        "risk_probability": round(float(prob), 4),
        "risk_label": label,
        "top_risk_factors": [],
        "model_version": "1.0.0",
    }
