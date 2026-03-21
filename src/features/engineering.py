import pandas as pd


def add_age_group(df: pd.DataFrame, age_col: str = "Age") -> pd.DataFrame:
    if age_col not in df.columns:
        age_col = "age"
    if age_col not in df.columns:
        return df

    df = df.copy()
    bins = [0, 18, 35, 55, 120]
    labels = ["Child", "Young", "Middle", "Senior"]
    df["age_group"] = pd.cut(df[age_col], bins=bins, labels=labels)
    return df


def add_bmi_category(df: pd.DataFrame) -> pd.DataFrame:
    if "BMI" not in df.columns:
        return df

    df = df.copy()
    bins = [0, 18.5, 25.0, 30.0, 100.0]
    labels = ["Underweight", "Normal", "Overweight", "Obese"]
    df["bmi_category"] = pd.cut(df["BMI"], bins=bins, labels=labels)
    return df


def add_glucose_insulin_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if "Glucose" not in df.columns or "Insulin" not in df.columns:
        return df

    df = df.copy()
    df["glucose_insulin_ratio"] = df["Glucose"] / (df["Insulin"] + 1)
    return df


def add_cardiac_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    required = ["trestbps", "chol", "thalach"]
    if not all(col in df.columns for col in required):
        return df

    df = df.copy()
    trestbps_norm = (df["trestbps"] - df["trestbps"].mean()) / (
        df["trestbps"].std() + 1e-8
    )
    chol_norm = (df["chol"] - df["chol"].mean()) / (df["chol"].std() + 1e-8)
    thalach_inv_norm = (
        (1 / (df["thalach"] + 1)) - (1 / (df["thalach"] + 1)).mean()
    ) / ((1 / (df["thalach"] + 1)).std() + 1e-8)
    df["cardiac_risk_score"] = trestbps_norm + chol_norm + thalach_inv_norm
    return df


def add_liver_enzyme_ratio(df: pd.DataFrame) -> pd.DataFrame:
    if (
        "Alamine_Aminotransferase" not in df.columns
        or "Aspartate_Aminotransferase" not in df.columns
    ):
        return df

    df = df.copy()
    df["liver_enzyme_ratio"] = df["Alamine_Aminotransferase"] / (
        df["Aspartate_Aminotransferase"] + 1
    )
    return df


def engineer_features(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    df = add_age_group(df)

    if dataset_name == "heart":
        df = add_cardiac_risk_score(df)
    elif dataset_name == "diabetes":
        df = add_bmi_category(df)
        df = add_glucose_insulin_ratio(df)
    elif dataset_name == "liver":
        df = add_liver_enzyme_ratio(df)

    return df
