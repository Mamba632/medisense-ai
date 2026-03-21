import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from src.config import DATASETS


class OutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        q1 = X.quantile(0.25)
        q3 = X.quantile(0.75)
        iqr = q3 - q1
        self.lower_ = q1 - self.factor * iqr
        self.upper_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].clip(lower=self.lower_[col], upper=self.upper_[col])
        return X.values

    def get_feature_names_out(self, input_features=None):
        return input_features


def build_preprocessor(dataset_name: str) -> ColumnTransformer:
    config = DATASETS[dataset_name]
    numeric_features = config["numeric_features"]
    categorical_features = config["categorical_features"]

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clipper", OutlierClipper(factor=1.5)),
            ("scaler", StandardScaler()),
        ]
    )

    transformers = [
        ("num", numeric_pipeline, numeric_features),
    ]

    if categorical_features:
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        drop="first", sparse_output=False, handle_unknown="ignore"
                    ),
                ),
            ]
        )
        transformers.append(("cat", categorical_pipeline, categorical_features))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )

    return preprocessor


class MediSensePreprocessor:
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.pipeline = build_preprocessor(dataset_name)
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y=None):
        self.pipeline.fit(X, y)
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before transform.")
        return self.pipeline.transform(X)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names_out(self) -> list[str]:
        return list(self.pipeline.get_feature_names_out())
