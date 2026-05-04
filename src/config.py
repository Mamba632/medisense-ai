import os
from pathlib import Path

# Project root
ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_CACHE_DIR = ROOT_DIR / ".cache"
LOCAL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE_DIR / "matplotlib"))

# Directories
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SPLITS_DIR = DATA_DIR / "splits"
MODELS_DIR = ROOT_DIR / "models_saved"
NOTEBOOKS_DIR = ROOT_DIR / "notebooks"

# Random seed for reproducibility
SEED = 42

# Train/Val/Test split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Dataset configurations
DATASETS = {
    "heart": {
        "filename": "heart.csv",
        "target": "target",
        "numeric_features": [
            "age",
            "trestbps",
            "chol",
            "thalach",
            "oldpeak",
        ],
        "categorical_features": [
            "sex",
            "cp",
            "fbs",
            "restecg",
            "exang",
            "slope",
            "ca",
            "thal",
        ],
        "description": "Heart Disease (Cleveland) — UCI/Kaggle",
    },
    "diabetes": {
        "filename": "diabetes.csv",
        "target": "Outcome",
        "numeric_features": [
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age",
        ],
        "categorical_features": [],
        "description": "PIMA Indians Diabetes — Kaggle/UCI",
    },
    "liver": {
        "filename": "indian_liver_patient.csv",
        "target": "Dataset",
        "numeric_features": [
            "Age",
            "Total_Bilirubin",
            "Direct_Bilirubin",
            "Alkaline_Phosphotase",
            "Alamine_Aminotransferase",
            "Aspartate_Aminotransferase",
            "Total_Protiens",
            "Albumin",
            "Albumin_and_Globulin_Ratio",
        ],
        "categorical_features": ["Gender"],
        "description": "Indian Liver Patient Dataset — UCI/Kaggle",
    },
}

# Base learner hyperparameters
BASE_LEARNER_PARAMS = {
    "random_forest": {
        "n_estimators": 200,
        "max_depth": None,
        "random_state": SEED,
    },
    "xgboost": {
        "n_estimators": 200,
        "tree_method": "hist",
        "random_state": SEED,
        "eval_metric": "logloss",
    },
    "lightgbm": {
        "n_estimators": 200,
        "boosting_type": "gbdt",
        "random_state": SEED,
        "verbose": -1,
    },
    "svm": {
        "kernel": "rbf",
        "probability": True,
        "random_state": SEED,
    },
    "knn": {
        "n_neighbors": 7,
    },
}

# Deep learning hyperparameters
DL_PARAMS = {
    "hidden_layers": [256, 128, 64],
    "dropout_rate": 0.3,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 200,
    "patience": 10,
}

# Stacking ensemble
STACKING_PARAMS = {
    "n_folds": 5,
    "meta_learner_C": 1.0,
}

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
