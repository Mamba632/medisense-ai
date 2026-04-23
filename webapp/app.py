from __future__ import annotations

# ruff: noqa: E402

import hmac
import json
import os
import secrets
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import (
    Flask,
    Response,
    abort,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)
from flask_login import (
    LoginManager,
    UserMixin,
    current_user,
    login_required,
    login_user,
    logout_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import check_password_hash, generate_password_hash

from src.config import DATASETS
from src.services import prediction_service
from src.services.prediction_service import (
    ModelArtifactError,
    PredictionValidationError,
    validate_prediction_input,
)
from webapp.forms import (
    ASSESSMENT_TIPS,
    DISEASE_INFO,
    DROPDOWN_OPTIONS,
    FEATURE_INPUT_META,
    FEATURE_LABELS,
    NEXT_STEPS,
    REPORT_CHECKLIST,
)

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - only relevant for minimal deployments
    load_dotenv = None

try:
    from flask_migrate import Migrate
except ImportError:  # pragma: no cover - migration support is optional at import time
    Migrate = None

if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


db = SQLAlchemy()
login_manager = LoginManager()
migrate = Migrate() if Migrate else None

LOGIN_WINDOW_SECONDS = 300
LOGIN_MAX_FAILURES = 5
_login_failures: dict[str, list[float]] = {}

UNSAFE_SECRET_VALUES = {
    "change-this-secret-key",
    "generate-a-long-random-secret",
    "medisense-ai-secret-key-2026",
}
LOCAL_DEV_SECRET_KEY = "medisense-ai-local-development-secret-key"


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    predictions = db.relationship("Prediction", backref="user", lazy=True)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    disease_type = db.Column(db.String(50), nullable=False)
    risk_probability = db.Column(db.Float, nullable=False)
    risk_label = db.Column(db.String(20), nullable=False)
    input_data = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, server_default=db.func.now())


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(__name__)
    db_path = Path(__file__).parent / "medisense.db"

    app.config.from_mapping(
        SECRET_KEY=os.environ.get("SECRET_KEY"),
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", f"sqlite:///{db_path}"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        AUTO_CREATE_TABLES=False,
    )

    if test_config:
        app.config.update(test_config)

    app.secret_key = _resolve_secret_key(app)
    db.init_app(app)
    if migrate:
        migrate.init_app(app, db)

    login_manager.login_view = "login"
    login_manager.login_message_category = "info"
    login_manager.init_app(app)

    _register_security(app)
    _register_template_helpers(app)
    _register_routes(app)
    _register_cli(app)

    if app.config.get("AUTO_CREATE_TABLES"):
        init_database(app)

    return app


def init_database(app: Flask) -> None:
    with app.app_context():
        db.create_all()


def _register_routes(app: Flask) -> None:
    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/signup", methods=["GET", "POST"])
    def signup():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            name = request.form.get("name", "").strip()
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            confirm = request.form.get("confirm_password", "")

            if not name or not email or not password:
                flash("All fields are required.", "danger")
                return render_template("signup.html")

            if password != confirm:
                flash("Passwords do not match.", "danger")
                return render_template("signup.html")

            if len(password) < 6:
                flash("Password must be at least 6 characters.", "danger")
                return render_template("signup.html")

            existing = User.query.filter_by(email=email).first()
            if existing:
                flash("Email already registered. Please log in.", "warning")
                return redirect(url_for("login"))

            user = User(
                name=name,
                email=email,
                password_hash=generate_password_hash(password),
            )
            db.session.add(user)
            db.session.commit()
            flash("Account created successfully! Please log in.", "success")
            return redirect(url_for("login"))

        return render_template("signup.html")

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if current_user.is_authenticated:
            return redirect(url_for("dashboard"))

        if request.method == "POST":
            email = request.form.get("email", "").strip().lower()
            password = request.form.get("password", "")
            key = _login_rate_key(email)

            if _is_login_rate_limited(key):
                flash(
                    "Too many failed login attempts. Please wait a few minutes.",
                    "danger",
                )
                return render_template("login.html"), 429

            user = User.query.filter_by(email=email).first()
            if user and check_password_hash(user.password_hash, password):
                _clear_login_failures(key)
                login_user(user)
                flash(f"Welcome back, {user.name}!", "success")
                next_page = request.args.get("next")
                if next_page and _is_safe_url(next_page):
                    return redirect(next_page)
                return redirect(url_for("dashboard"))

            _record_login_failure(key)
            flash("Invalid email or password.", "danger")

        return render_template("login.html")

    @app.route("/logout", methods=["POST"])
    @login_required
    def logout():
        logout_user()
        flash("You have been logged out.", "info")
        return redirect(url_for("index"))

    @app.route("/dashboard")
    @login_required
    def dashboard():
        all_predictions = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
        recent_predictions = all_predictions[:5]
        highest_prediction = (
            max(all_predictions, key=lambda pred: pred.risk_probability)
            if all_predictions
            else None
        )
        dashboard_stats = {
            "total": len(all_predictions),
            "tracked_conditions": len({pred.disease_type for pred in all_predictions}),
            "medium_high": sum(
                1 for pred in all_predictions if pred.risk_label in {"Medium", "High"}
            ),
            "highest_probability": highest_prediction.risk_probability
            if highest_prediction
            else None,
            "highest_disease": highest_prediction.disease_type
            if highest_prediction
            else None,
            "last_assessment": all_predictions[0].created_at
            if all_predictions
            else None,
        }
        return render_template(
            "dashboard.html",
            diseases=DISEASE_INFO,
            predictions=recent_predictions,
            stats=dashboard_stats,
        )

    @app.route("/predict/<disease>", methods=["GET", "POST"])
    @login_required
    def predict(disease):
        if disease not in DATASETS:
            flash("Invalid disease type.", "danger")
            return redirect(url_for("dashboard"))

        features = _features_for_disease(disease)

        if request.method == "POST":
            raw_input = {feature: request.form.get(feature, "") for feature in features}

            try:
                validated_input = validate_prediction_input(disease, raw_input)
            except PredictionValidationError as exc:
                flash("Please fix the highlighted fields.", "danger")
                return _render_prediction_form(
                    disease,
                    field_errors=exc.field_errors,
                    form_values=raw_input,
                )

            try:
                result = prediction_service.predict_risk(disease, validated_input)
            except (ModelArtifactError, RuntimeError) as exc:
                flash(f"Prediction failed: {exc}", "danger")
                return _render_prediction_form(
                    disease,
                    form_values=raw_input,
                    status_code=500,
                )

            prediction = Prediction(
                user_id=current_user.id,
                disease_type=disease,
                risk_probability=result["risk_probability"],
                risk_label=result["risk_label"],
                input_data=json.dumps(validated_input),
            )
            db.session.add(prediction)
            db.session.commit()

            return render_template(
                "result.html",
                disease_info=DISEASE_INFO[disease],
                disease=disease,
                probability=result["risk_probability"],
                risk_label=result["risk_label"],
                input_data=validated_input,
                feature_labels=FEATURE_LABELS,
                next_steps=NEXT_STEPS[result["risk_label"]],
                report_checklist=REPORT_CHECKLIST,
                top_risk_factors=result["top_risk_factors"],
                calibrated=result.get("calibrated", False),
                prediction_id=prediction.id,
            )

        return _render_prediction_form(disease)

    @app.route("/history")
    @login_required
    def history():
        predictions = (
            Prediction.query.filter_by(user_id=current_user.id)
            .order_by(Prediction.created_at.desc())
            .all()
        )
        return render_template(
            "history.html",
            predictions=predictions,
            disease_info=DISEASE_INFO,
        )

    @app.route("/prediction/<int:prediction_id>/report")
    @login_required
    def download_report(prediction_id: int):
        prediction = Prediction.query.filter_by(
            id=prediction_id, user_id=current_user.id
        ).first_or_404()
        report = _build_doctor_report(prediction)
        filename = f"medisense-{prediction.disease_type}-report-{prediction.id}.txt"
        return Response(
            report,
            mimetype="text/plain",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )


def _render_prediction_form(
    disease: str,
    field_errors: dict[str, str] | None = None,
    form_values: dict[str, str] | None = None,
    status_code: int = 200,
):
    response = render_template(
        "predict.html",
        disease=disease,
        disease_info=DISEASE_INFO[disease],
        features=_features_for_disease(disease),
        feature_labels=FEATURE_LABELS,
        feature_meta=FEATURE_INPUT_META,
        dropdown_options=DROPDOWN_OPTIONS,
        assessment_tips=ASSESSMENT_TIPS.get(disease, []),
        field_errors=field_errors or {},
        form_values=form_values or {},
    )
    return response, status_code


def _features_for_disease(disease: str) -> list[str]:
    config = DATASETS[disease]
    return config["numeric_features"] + config["categorical_features"]


def _build_doctor_report(prediction: Prediction) -> str:
    input_data = json.loads(prediction.input_data)
    disease = DISEASE_INFO[prediction.disease_type]["title"]
    created_at = (
        prediction.created_at.strftime("%d %b %Y, %H:%M")
        if prediction.created_at
        else "N/A"
    )

    lines = [
        "MediSense AI Doctor-Friendly Screening Summary",
        "",
        f"Disease: {disease}",
        f"Assessment date: {created_at}",
        f"Risk label: {prediction.risk_label}",
        f"Risk score: {prediction.risk_probability * 100:.1f}%",
        "",
        "Important: This is an AI screening estimate, not a diagnosis.",
        "Clinical decisions should be made with a qualified healthcare professional.",
        "",
        "Submitted health indicators:",
    ]

    for key, value in input_data.items():
        label = FEATURE_LABELS.get(key, key)
        lines.append(f"- {label}: {value}")

    lines.extend(
        [
            "",
            "Bring to appointment:",
            *[f"- {item}" for item in REPORT_CHECKLIST],
            "",
        ]
    )
    return "\n".join(lines)


def _register_security(app: Flask) -> None:
    @app.before_request
    def csrf_protect():
        if request.method != "POST":
            return None

        expected = session.get("_csrf_token")
        supplied = request.form.get("_csrf_token") or request.headers.get(
            "X-CSRF-Token"
        )
        if not expected or not supplied or not hmac.compare_digest(expected, supplied):
            abort(400, description="Invalid CSRF token.")
        return None


def _register_template_helpers(app: Flask) -> None:
    @app.context_processor
    def inject_helpers():
        return {"csrf_token": _csrf_token}


def _register_cli(app: Flask) -> None:
    @app.cli.command("init-db")
    def init_db_command():
        init_database(app)
        print("Initialized MediSense database.")


def _csrf_token() -> str:
    token = session.get("_csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["_csrf_token"] = token
    return token


def _resolve_secret_key(app: Flask) -> str:
    secret = app.config.get("SECRET_KEY")

    if app.config.get("TESTING"):
        return secret or "test-secret-key"

    if secret and secret not in UNSAFE_SECRET_VALUES:
        return secret

    if _is_local_development():
        return LOCAL_DEV_SECRET_KEY

    if not secret:
        raise RuntimeError(
            "SECRET_KEY environment variable is required. "
            "Copy .env.example to .env and set a unique random value."
        )

    raise RuntimeError("SECRET_KEY must be changed before running the app.")


def _is_local_development() -> bool:
    environment = os.environ.get(
        "MEDISENSE_ENV", os.environ.get("FLASK_ENV", "development")
    )
    return environment.lower() not in {"prod", "production"}


def _is_safe_url(target: str) -> bool:
    host_url = request.host_url
    ref_url = urlparse(host_url)
    test_url = urlparse(urljoin(host_url, target))
    return test_url.scheme in {"http", "https"} and ref_url.netloc == test_url.netloc


def _login_rate_key(email: str) -> str:
    remote = request.headers.get("X-Forwarded-For", request.remote_addr or "local")
    remote = remote.split(",", 1)[0].strip()
    return f"{remote}:{email.lower()}"


def _is_login_rate_limited(key: str) -> bool:
    failures = _recent_login_failures(key)
    _login_failures[key] = failures
    return len(failures) >= LOGIN_MAX_FAILURES


def _record_login_failure(key: str) -> None:
    failures = _recent_login_failures(key)
    failures.append(time.monotonic())
    _login_failures[key] = failures


def _clear_login_failures(key: str) -> None:
    _login_failures.pop(key, None)


def _recent_login_failures(key: str) -> list[float]:
    cutoff = time.monotonic() - LOGIN_WINDOW_SECONDS
    return [
        timestamp for timestamp in _login_failures.get(key, []) if timestamp >= cutoff
    ]


app = create_app()


if __name__ == "__main__":
    init_database(app)
    flask_host = os.environ.get("FLASK_HOST", "127.0.0.1")
    flask_port = int(os.environ.get("FLASK_PORT", "5000"))
    flask_debug = os.environ.get("FLASK_DEBUG", "1").lower() in {"1", "true", "yes"}
    app.run(host=flask_host, port=flask_port, debug=flask_debug)
