import pandas as pd
from src.config import RAW_DATA_DIR, DATASETS


def load_raw_dataset(name: str) -> pd.DataFrame:
    if name not in DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}"
        )

    config = DATASETS[name]
    filepath = RAW_DATA_DIR / config["filename"]

    if not filepath.exists():
        raise FileNotFoundError(f"Dataset file not found: {filepath}")

    df = pd.read_csv(filepath)
    return df


def load_clean_dataset(name: str) -> pd.DataFrame:
    df = load_raw_dataset(name)
    target = DATASETS[name]["target"]

    # Replace '?' with NaN and coerce numeric columns
    df = df.replace("?", float("nan"))
    for col in df.columns:
        if col != "Gender":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Binarize targets
    if name == "heart":
        df[target] = (df[target].astype(int) > 0).astype(int)
    elif name == "liver":
        df[target] = df[target].map({1: 1, 2: 0})

    return df


def get_feature_columns(name: str) -> tuple[list[str], list[str]]:
    config = DATASETS[name]
    return config["numeric_features"], config["categorical_features"]


def get_target_column(name: str) -> str:
    return DATASETS[name]["target"]
