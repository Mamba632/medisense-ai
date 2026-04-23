"""
Tests for compute_metrics (src/evaluation/metrics.py).
"""

import numpy as np
import pytest

from src.evaluation.metrics import compute_metrics


@pytest.fixture
def known_labels():
    """Simple known ground truth and predictions."""
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0])
    return y_true, y_pred


@pytest.fixture
def known_probs():
    """Probability scores corresponding to the known labels."""
    return np.array([0.1, 0.6, 0.9, 0.8, 0.3, 0.2, 0.85, 0.15])


class TestComputeMetrics:
    def test_with_known_inputs(self, known_labels):
        """Metrics should be computable with valid y_true and y_pred."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred)
        # Accuracy: 6 correct out of 8 = 0.75
        assert abs(metrics["accuracy"] - 0.75) < 1e-6

    def test_all_expected_keys_present(self, known_labels):
        """Without y_prob, result should have accuracy, precision, recall, f1, confusion_matrix."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred)
        expected_keys = {"accuracy", "precision", "recall", "f1", "confusion_matrix"}
        assert expected_keys.issubset(metrics.keys()), (
            f"Missing keys: {expected_keys - metrics.keys()}"
        )

    def test_confusion_matrix_shape(self, known_labels):
        """Confusion matrix should be a 2x2 nested list."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred)
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2 and len(cm[0]) == 2

    def test_with_y_prob_adds_roc_auc_and_pr_auc(self, known_labels, known_probs):
        """When y_prob is provided, roc_auc and pr_auc should be in the result."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred, y_prob=known_probs)
        assert "roc_auc" in metrics, "roc_auc missing when y_prob is provided"
        assert "pr_auc" in metrics, "pr_auc missing when y_prob is provided"

    def test_roc_auc_is_valid(self, known_labels, known_probs):
        """roc_auc should be a float between 0 and 1."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred, y_prob=known_probs)
        assert 0.0 <= metrics["roc_auc"] <= 1.0
        assert 0.0 <= metrics["pr_auc"] <= 1.0

    def test_metrics_values_are_floats(self, known_labels):
        """Accuracy, precision, recall, and f1 should all be floats."""
        y_true, y_pred = known_labels
        metrics = compute_metrics(y_true, y_pred)
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert isinstance(metrics[key], float), f"{key} is not a float"
