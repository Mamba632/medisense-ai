"""
Tests for the FastAPI application (src/api/main.py).

Note: Actual prediction endpoints require trained models in models_saved/,
so we only test routes that do not depend on model artifacts.
"""

from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_200(self):
        """GET /health should return HTTP 200."""
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_body(self):
        """GET /health should indicate healthy status."""
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"


class TestModelsEndpoint:
    def test_models_returns_200(self):
        """GET /models should return HTTP 200."""
        response = client.get("/models")
        assert response.status_code == 200

    def test_models_returns_list(self):
        """GET /models should return a list of model names."""
        response = client.get("/models")
        data = response.json()
        assert "models" in data
        assert isinstance(data["models"], list)
        assert len(data["models"]) > 0

    def test_models_contains_expected_diseases(self):
        """The models list should contain heart, diabetes, and liver."""
        response = client.get("/models")
        models = response.json()["models"]
        for disease in ["heart", "diabetes", "liver"]:
            assert disease in models, f"'{disease}' not found in models list"


class TestFeaturesEndpoint:
    def test_heart_features_returns_200(self):
        """GET /predict/heart/features should return HTTP 200."""
        response = client.get("/predict/heart/features")
        assert response.status_code == 200

    def test_heart_features_has_schema(self):
        """The features response should contain feature properties."""
        response = client.get("/predict/heart/features")
        data = response.json()
        assert data["disease"] == "heart"
        assert "features" in data
        assert isinstance(data["features"], dict)
        assert len(data["features"]) > 0


class TestPredictUnknownDisease:
    def test_unknown_disease_returns_404(self):
        """POST /predict/unknown should return HTTP 404."""
        response = client.post("/predict/unknown", json={"age": 50})
        assert response.status_code == 404

    def test_unknown_disease_error_detail(self):
        """The 404 response should contain a meaningful error detail."""
        response = client.post("/predict/unknown", json={"age": 50})
        data = response.json()
        assert "detail" in data
