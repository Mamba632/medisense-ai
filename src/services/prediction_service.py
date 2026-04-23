from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from pydantic import ValidationError

from src.api.schemas import INPUT_SCHEMAS
from src.config import DATASETS, MODELS_DIR

MODEL_VERSION = "1.0.0"
CALIBRATOR_FILENAME = "probability_calibrator.pkl"
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

_artifact_cache: dict[str, dict[str, Any]] = {}


@dataclass
class PredictionResult:
    disease: str
    risk_probability: float
    risk_label: str
    top_risk_factors: list[dict[str, Any]]
    model_version: str = MODEL_VERSION
    calibrated: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "disease": self.disease,
            "risk_probability": round(float(self.risk_probability), 4),
            "risk_label": self.risk_label,
            "top_risk_factors": self.top_risk_factors,
            "model_version": self.model_version,
            "calibrated": self.calibrated,
        }


class PredictionServiceError(RuntimeError):
    """Base error for prediction service failures."""


class ModelArtifactError(PredictionServiceError):
    """Raised when required model artifacts are missing or cannot be loaded."""


class PredictionValidationError(PredictionServiceError):
    def __init__(self, field_errors: dict[str, str]):
        super().__init__("Prediction input validation failed.")
        self.field_errors = field_errors


def clear_artifact_cache() -> None:
    _artifact_cache.clear()


def risk_label_for_probability(probability: float) -> str:
    if probability < 0.3:
        return "Low"
    if probability < 0.7:
        return "Medium"
    return "High"


def validate_prediction_input(
    dataset_name: str, raw_data: dict[str, Any]
) -> dict[str, Any]:
    schema_cls = INPUT_SCHEMAS.get(dataset_name)
    if schema_cls is None:
        raise PredictionValidationError({"disease": "Unknown disease type."})

    try:
        return schema_cls(**raw_data).model_dump()
    except ValidationError as exc:
        field_errors: dict[str, str] = {}
        for err in exc.errors():
            field = str(err["loc"][0]) if err.get("loc") else "input"
            field_errors[field] = _friendly_validation_message(err)
        raise PredictionValidationError(field_errors) from exc


def predict_risk(dataset_name: str, input_data: dict[str, Any]) -> dict[str, Any]:
    if dataset_name not in DATASETS:
        raise PredictionServiceError(f"Unknown dataset: {dataset_name}")

    artifacts = load_artifacts(dataset_name)
    preprocessor = artifacts["preprocessor"]
    ensemble = artifacts["ensemble"]

    input_df = pd.DataFrame([input_data])
    transformed = preprocessor.transform(input_df)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Could not find the number of physical cores.*",
            category=UserWarning,
        )
        raw_probability = float(ensemble.predict_proba(transformed)[0, 1])
    probability, calibrated = _apply_calibration(raw_probability, artifacts)

    result = PredictionResult(
        disease=dataset_name,
        risk_probability=probability,
        risk_label=risk_label_for_probability(probability),
        top_risk_factors=_top_input_signals(preprocessor, transformed),
        calibrated=calibrated,
    )
    return result.to_dict()


def load_artifacts(dataset_name: str) -> dict[str, Any]:
    if dataset_name in _artifact_cache:
        return _artifact_cache[dataset_name]

    model_dir = MODELS_DIR / dataset_name
    ensemble_path = model_dir / "stacking_ensemble.pkl"
    preprocessor_path = model_dir / "preprocessor.pkl"
    calibrator_path = model_dir / CALIBRATOR_FILENAME

    try:
        artifacts = {
            "ensemble": joblib.load(ensemble_path),
            "preprocessor": joblib.load(preprocessor_path),
            "calibrator": _load_optional_calibrator(calibrator_path),
        }
    except ModuleNotFoundError as exc:
        missing = getattr(exc, "name", None) or str(exc)
        raise ModelArtifactError(
            "Model dependency is missing: "
            f"{missing}. Run 'pip install -r requirements.txt' before predicting."
        ) from exc
    except FileNotFoundError as exc:
        raise ModelArtifactError(
            f"Model artifact not found for '{dataset_name}'. "
            "Run 'python -m scripts.train_all --dataset all' first."
        ) from exc

    _artifact_cache[dataset_name] = artifacts
    return artifacts


def _load_optional_calibrator(path: Path) -> Any | None:
    if not path.exists():
        return None
    return joblib.load(path)


def _apply_calibration(
    raw_probability: float, artifacts: dict[str, Any]
) -> tuple[float, bool]:
    calibrator = artifacts.get("calibrator")
    if calibrator is None:
        return raw_probability, False

    raw = np.array([[raw_probability]])
    if hasattr(calibrator, "predict_proba"):
        calibrated = float(calibrator.predict_proba(raw)[0, 1])
    else:
        calibrated = float(calibrator.predict(raw)[0])

    return min(max(calibrated, 0.0), 1.0), True


def _top_input_signals(
    preprocessor: Any, transformed: Any, limit: int = 3
) -> list[dict]:
    values = np.asarray(transformed).reshape(-1)
    if values.size == 0:
        return []

    names = _feature_names(preprocessor, values.size)
    ranked_indexes = np.argsort(np.abs(values))[::-1][:limit]
    signals = []

    for index in ranked_indexes:
        value = float(values[index])
        if abs(value) < 1e-9:
            continue

        display_name = _clean_feature_name(names[index])
        direction = "above baseline" if value > 0 else "below baseline"
        signals.append(
            {
                "feature": names[index],
                "display_name": display_name,
                "direction": direction,
                "strength": round(abs(value), 3),
                "explanation": (
                    f"{display_name} was one of the strongest input signals "
                    f"for this screening estimate ({direction})."
                ),
            }
        )

    return signals


def _feature_names(preprocessor: Any, fallback_size: int) -> list[str]:
    try:
        names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = [f"feature_{i + 1}" for i in range(fallback_size)]

    if len(names) < fallback_size:
        names.extend(f"feature_{i + 1}" for i in range(len(names), fallback_size))
    return names[:fallback_size]


def _clean_feature_name(name: str) -> str:
    cleaned = name.split("__", 1)[-1]
    cleaned = cleaned.replace("_", " ")
    return cleaned.strip().title()


def _friendly_validation_message(error: dict[str, Any]) -> str:
    message = str(error.get("msg", "Invalid value."))
    if message.startswith("Value error, "):
        return message.replace("Value error, ", "", 1)
    if "greater than or equal" in message:
        return message.replace("Input should be ", "Value must be ")
    if "less than or equal" in message:
        return message.replace("Input should be ", "Value must be ")
    if "valid number" in message:
        return "Enter a valid number."
    if "String should match pattern" in message:
        return "Choose one of the available options."
    return message
