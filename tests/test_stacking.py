"""
Tests for StackingEnsemble (src/models/stacking.py).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.models.stacking import StackingEnsemble


@pytest.fixture
def sample_data():
    """Generate a simple binary classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def ensemble():
    """Create a StackingEnsemble with lightweight base learners."""
    base_learners = [
        ("dt", DecisionTreeClassifier(random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=10, random_state=42)),
    ]
    meta_learner = LogisticRegression(random_state=42)
    return StackingEnsemble(base_learners, meta_learner, n_folds=3)


class TestStackingEnsemble:
    def test_fit_returns_self(self, ensemble, sample_data):
        """fit() should return the ensemble instance itself."""
        X, y = sample_data
        result = ensemble.fit(X, y)
        assert result is ensemble

    def test_predict_returns_correct_shape(self, ensemble, sample_data):
        """predict() output should have the same length as input samples."""
        X, y = sample_data
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        assert preds.shape == (X.shape[0],), (
            f"Expected shape ({X.shape[0]},), got {preds.shape}"
        )

    def test_predict_returns_binary_values(self, ensemble, sample_data):
        """predict() should return only 0s and 1s."""
        X, y = sample_data
        ensemble.fit(X, y)
        preds = ensemble.predict(X)
        unique_vals = set(np.unique(preds))
        assert unique_vals.issubset({0, 1}), f"Expected only 0 and 1, got {unique_vals}"

    def test_predict_proba_returns_probabilities(self, ensemble, sample_data):
        """predict_proba() values should be between 0 and 1."""
        X, y = sample_data
        ensemble.fit(X, y)
        proba = ensemble.predict_proba(X)
        assert proba.shape == (X.shape[0], 2), (
            f"Expected shape ({X.shape[0]}, 2), got {proba.shape}"
        )
        assert np.all(proba >= 0.0) and np.all(proba <= 1.0), (
            "Probabilities must be between 0 and 1"
        )
