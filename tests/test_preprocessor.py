"""
Tests for MediSensePreprocessor (src/data/preprocessor.py).
"""

import numpy as np
import pandas as pd
import pytest

from src.config import RAW_DATA_DIR
from src.data.preprocessor import MediSensePreprocessor


@pytest.fixture
def heart_df():
    """Load the heart dataset for testing."""
    df = pd.read_csv(RAW_DATA_DIR / "heart.csv")
    return df


@pytest.fixture
def heart_features(heart_df):
    """Return feature columns (everything except target)."""
    return heart_df.drop(columns=["target"])


@pytest.fixture
def preprocessor():
    return MediSensePreprocessor("heart")


class TestMediSensePreprocessor:
    def test_fit_transform_runs_without_error(self, preprocessor, heart_features):
        """fit_transform should complete without raising any exception."""
        result = preprocessor.fit_transform(heart_features)
        assert result is not None

    def test_output_has_no_nan(self, preprocessor, heart_features):
        """Transformed output must not contain any NaN values."""
        result = preprocessor.fit_transform(heart_features)
        assert not np.isnan(result).any(), "Output contains NaN values"

    def test_output_shape_is_correct(self, preprocessor, heart_features):
        """Number of rows must match input; columns depend on encoding."""
        result = preprocessor.fit_transform(heart_features)
        assert result.shape[0] == heart_features.shape[0], (
            f"Expected {heart_features.shape[0]} rows, got {result.shape[0]}"
        )
        # There should be at least as many columns as numeric features
        assert result.shape[1] >= 5, (
            f"Expected at least 5 columns, got {result.shape[1]}"
        )

    def test_get_feature_names_out_returns_list(self, preprocessor, heart_features):
        """get_feature_names_out should return a list of strings after fitting."""
        preprocessor.fit_transform(heart_features)
        names = preprocessor.get_feature_names_out()
        assert isinstance(names, list), f"Expected list, got {type(names)}"
        assert len(names) > 0, "Feature names list is empty"
        assert all(isinstance(n, str) for n in names), (
            "Not all feature names are strings"
        )
