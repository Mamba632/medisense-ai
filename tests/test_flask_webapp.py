import re

import pytest
from werkzeug.security import generate_password_hash

from src.api.schemas import INPUT_SCHEMAS
from src.config import DATASETS
from src.services import prediction_service
from webapp.app import Prediction, User, create_app, db


@pytest.fixture()
def app_instance():
    app = create_app(
        {
            "TESTING": True,
            "SECRET_KEY": "test-secret-key",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        }
    )
    with app.app_context():
        db.create_all()
    yield app
    with app.app_context():
        db.session.remove()
        db.drop_all()


@pytest.fixture()
def client(app_instance):
    return app_instance.test_client()


def _csrf_token(response):
    match = re.search(rb'name="_csrf_token" value="([^"]+)"', response.data)
    assert match, "CSRF token not found"
    return match.group(1).decode()


def _create_user(email="user@example.com", password="secret123", name="Test User"):
    user = User(
        name=name,
        email=email,
        password_hash=generate_password_hash(password),
    )
    db.session.add(user)
    db.session.commit()
    return user


def _login(client, email="user@example.com", password="secret123", next_url=None):
    path = "/login"
    if next_url:
        path = f"/login?next={next_url}"
    response = client.get(path)
    token = _csrf_token(response)
    return client.post(
        path,
        data={"email": email, "password": password, "_csrf_token": token},
        follow_redirects=False,
    )


def test_home_page_loads(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"MediSense AI" in response.data


def test_login_page_loads(client):
    response = client.get("/login")

    assert response.status_code == 200
    assert b"Login" in response.data


def test_dashboard_requires_login(client):
    response = client.get("/dashboard")

    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_signup_creates_user(client, app_instance):
    response = client.get("/signup")
    token = _csrf_token(response)

    response = client.post(
        "/signup",
        data={
            "name": "New User",
            "email": "new@example.com",
            "password": "secret123",
            "confirm_password": "secret123",
            "_csrf_token": token,
        },
        follow_redirects=False,
    )

    assert response.status_code == 302
    with app_instance.app_context():
        assert User.query.filter_by(email="new@example.com").first() is not None


def test_login_rejects_wrong_password(client, app_instance):
    with app_instance.app_context():
        _create_user()

    response = _login(client, password="wrong")

    assert response.status_code == 200
    assert b"Invalid email or password" in response.data


def test_login_next_rejects_external_redirect(client, app_instance):
    with app_instance.app_context():
        _create_user()

    response = _login(client, next_url="https://example.com/phish")

    assert response.status_code == 302
    assert response.headers["Location"].endswith("/dashboard")


def test_create_app_uses_local_dev_secret_without_env(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.delenv("FLASK_ENV", raising=False)
    monkeypatch.setenv("MEDISENSE_ENV", "development")

    app = create_app(
        {
            "TESTING": False,
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        }
    )

    assert app.secret_key == "medisense-ai-local-development-secret-key"


def test_create_app_requires_real_secret_in_production(monkeypatch):
    monkeypatch.delenv("SECRET_KEY", raising=False)
    monkeypatch.setenv("MEDISENSE_ENV", "production")

    with pytest.raises(RuntimeError, match="SECRET_KEY environment variable"):
        create_app(
            {
                "TESTING": False,
                "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            }
        )


def test_user_cannot_see_another_users_history(client, app_instance):
    with app_instance.app_context():
        first = _create_user("first@example.com", name="First")
        second = _create_user("second@example.com", name="Second")
        db.session.add(
            Prediction(
                user_id=first.id,
                disease_type="heart",
                risk_probability=0.2,
                risk_label="Low",
                input_data="{}",
            )
        )
        db.session.add(
            Prediction(
                user_id=second.id,
                disease_type="diabetes",
                risk_probability=0.9,
                risk_label="High",
                input_data="{}",
            )
        )
        db.session.commit()

    _login(client, email="first@example.com")
    response = client.get("/history")

    assert response.status_code == 200
    assert b"Heart Disease" in response.data
    assert b"Diabetes" not in response.data


def test_prediction_validation_rejects_impossible_values(client, app_instance):
    with app_instance.app_context():
        _create_user()

    _login(client)
    response = client.get("/predict/diabetes")
    token = _csrf_token(response)
    response = client.post(
        "/predict/diabetes",
        data={
            "_csrf_token": token,
            "Pregnancies": "1",
            "Glucose": "110",
            "BloodPressure": "72",
            "SkinThickness": "25",
            "Insulin": "80",
            "BMI": "24.5",
            "DiabetesPedigreeFunction": "0.45",
            "Age": "150",
        },
    )

    assert response.status_code == 200
    assert b"Please fix the highlighted fields" in response.data
    assert b"Age" in response.data


def test_heart_form_uses_trained_category_codes(client, app_instance):
    with app_instance.app_context():
        _create_user()

    _login(client)
    response = client.get("/predict/heart")

    assert response.status_code == 200
    assert b'value="4"' in response.data
    assert b"Asymptomatic" in response.data
    assert b'value="7"' in response.data
    assert b"Reversible Defect" in response.data


def test_heart_validation_rejects_untrained_category_code():
    valid_input = {
        "age": 52,
        "sex": 1,
        "cp": 4,
        "trestbps": 120,
        "chol": 200,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.2,
        "slope": 2,
        "ca": 0,
        "thal": 4,
    }

    with pytest.raises(prediction_service.PredictionValidationError) as exc_info:
        prediction_service.validate_prediction_input("heart", valid_input)

    assert exc_info.value.field_errors["thal"] == (
        "Choose one of the available options: 3, 6, 7."
    )


def test_prediction_route_saves_mocked_result(client, app_instance, monkeypatch):
    with app_instance.app_context():
        _create_user()

    def fake_predict(disease, input_data):
        return {
            "disease": disease,
            "risk_probability": 0.42,
            "risk_label": "Medium",
            "top_risk_factors": [
                {
                    "display_name": "Glucose",
                    "strength": 1.2,
                    "explanation": "Glucose was one of the strongest input signals.",
                }
            ],
            "model_version": "test",
            "calibrated": True,
        }

    monkeypatch.setattr(prediction_service, "predict_risk", fake_predict)

    _login(client)
    response = client.get("/predict/diabetes")
    token = _csrf_token(response)
    response = client.post(
        "/predict/diabetes",
        data={
            "_csrf_token": token,
            "Pregnancies": "1",
            "Glucose": "110",
            "BloodPressure": "72",
            "SkinThickness": "25",
            "Insulin": "80",
            "BMI": "24.5",
            "DiabetesPedigreeFunction": "0.45",
            "Age": "45",
        },
    )

    assert response.status_code == 200
    assert b"Medium Risk" in response.data
    assert b"Top Input Signals" in response.data

    with app_instance.app_context():
        prediction = Prediction.query.one()
        assert prediction.risk_label == "Medium"
        assert prediction.risk_probability == 0.42


def test_flask_features_match_pydantic_schemas():
    for disease, config in DATASETS.items():
        expected = set(config["numeric_features"] + config["categorical_features"])
        schema_fields = set(INPUT_SCHEMAS[disease].model_fields)
        assert expected == schema_fields
