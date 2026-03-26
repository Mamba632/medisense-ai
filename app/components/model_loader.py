import streamlit as st
import joblib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import MODELS_DIR
from src.data.loader import load_clean_dataset


@st.cache_resource
def load_model(dataset_name: str):
    model_path = MODELS_DIR / dataset_name / "stacking_ensemble.pkl"
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_resource
def load_preprocessor(dataset_name: str):
    path = MODELS_DIR / dataset_name / "preprocessor.pkl"
    if not path.exists():
        return None
    return joblib.load(path)


@st.cache_data
def load_data(dataset_name: str):
    return load_clean_dataset(dataset_name)


def is_model_available(dataset_name: str) -> bool:
    model_path = MODELS_DIR / dataset_name / "stacking_ensemble.pkl"
    return model_path.exists()
