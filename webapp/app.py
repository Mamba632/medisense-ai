import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_login import (
    LoginManager,
    login_user,
    logout_user,
    login_required,
    current_user,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import joblib
import pandas as pd

from src.config import DATASETS, MODELS_DIR

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "medisense-ai-secret-key-2026")

# Database
db_path = Path(__file__).parent / "medisense.db"
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"
login_manager.login_message_category = "info"


# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    predictions = db.relationship("Prediction", backref="user", lazy=True)

    @property
    def is_active(self):
        return True

    @property
    def is_authenticated(self):
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        return str(self.id)


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


# Cache loaded models
_model_cache = {}


def get_model(disease):
    if disease not in _model_cache:
        model_dir = MODELS_DIR / disease
        _model_cache[disease] = {
            "ensemble": joblib.load(model_dir / "stacking_ensemble.pkl"),
            "preprocessor": joblib.load(model_dir / "preprocessor.pkl"),
        }
    return _model_cache[disease]


# Disease info
DISEASE_INFO = {
    "heart": {
        "title": "Heart Disease",
        "icon": "❤️",
        "color": "#E63946",
        "description": "Predict risk of heart disease based on clinical indicators like blood pressure, cholesterol, and heart rate.",
    },
    "diabetes": {
        "title": "Diabetes",
        "icon": "🩸",
        "color": "#F4A261",
        "description": "Assess diabetes risk using indicators such as glucose level, BMI, insulin, and family history.",
    },
    "liver": {
        "title": "Liver Disease",
        "icon": "🫁",
        "color": "#2A9D8F",
        "description": "Evaluate liver disease risk based on enzyme levels, bilirubin, and protein indicators.",
    },
}

# Feature display labels
FEATURE_LABELS = {
    "age": "Age (years)",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure (mm Hg)",
    "chol": "Cholesterol (mg/dl)",
    "fbs": "Fasting Blood Sugar > 120 mg/dl",
    "restecg": "Resting ECG Results",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "Slope of Peak Exercise ST",
    "ca": "Number of Major Vessels (0-3)",
    "thal": "Thalassemia Type",
    "Pregnancies": "Number of Pregnancies",
    "Glucose": "Glucose Level (mg/dl)",
    "BloodPressure": "Blood Pressure (mm Hg)",
    "SkinThickness": "Skin Thickness (mm)",
    "Insulin": "Insulin Level (mu U/ml)",
    "BMI": "Body Mass Index",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age (years)",
    "Gender": "Gender",
    "Total_Bilirubin": "Total Bilirubin",
    "Direct_Bilirubin": "Direct Bilirubin",
    "Alkaline_Phosphotase": "Alkaline Phosphatase",
    "Alamine_Aminotransferase": "ALT (SGPT)",
    "Aspartate_Aminotransferase": "AST (SGOT)",
    "Total_Protiens": "Total Proteins",
    "Albumin": "Albumin",
    "Albumin_and_Globulin_Ratio": "A/G Ratio",
}

# Dropdown options for categorical features
DROPDOWN_OPTIONS = {
    "sex": [("0", "Female"), ("1", "Male")],
    "cp": [
        ("0", "Typical Angina"),
        ("1", "Atypical Angina"),
        ("2", "Non-anginal Pain"),
        ("3", "Asymptomatic"),
    ],
    "fbs": [("0", "No"), ("1", "Yes")],
    "restecg": [("0", "Normal"), ("1", "ST-T Abnormality"), ("2", "LV Hypertrophy")],
    "exang": [("0", "No"), ("1", "Yes")],
    "slope": [("0", "Upsloping"), ("1", "Flat"), ("2", "Downsloping")],
    "Gender": [("Male", "Male"), ("Female", "Female")],
}


# Routes
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if current_user.is_authenticated:
        return redirect(url_for("dashboard"))

    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
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
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password_hash, password):
            login_user(user)
            flash(f"Welcome back, {user.name}!", "success")
            next_page = request.args.get("next")
            return redirect(next_page or url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
@login_required
def logout():
    logout_user()
    flash("You have been logged out.", "info")
    return redirect(url_for("index"))


@app.route("/dashboard")
@login_required
def dashboard():
    recent_predictions = (
        Prediction.query.filter_by(user_id=current_user.id)
        .order_by(Prediction.created_at.desc())
        .limit(5)
        .all()
    )
    return render_template(
        "dashboard.html",
        diseases=DISEASE_INFO,
        predictions=recent_predictions,
    )


@app.route("/predict/<disease>", methods=["GET", "POST"])
@login_required
def predict(disease):
    if disease not in DATASETS:
        flash("Invalid disease type.", "danger")
        return redirect(url_for("dashboard"))

    config = DATASETS[disease]
    features = config["numeric_features"] + config["categorical_features"]
    disease_info = DISEASE_INFO[disease]

    if request.method == "POST":
        input_data = {}
        for feature in features:
            val = request.form.get(feature, "0")
            if feature == "Gender":
                input_data[feature] = val
            else:
                input_data[feature] = float(val)

        # Predict
        artifacts = get_model(disease)
        input_df = pd.DataFrame([input_data])
        X = artifacts["preprocessor"].transform(input_df)
        prob = float(artifacts["ensemble"].predict_proba(X)[0, 1])

        if prob < 0.3:
            risk_label = "Low"
        elif prob < 0.7:
            risk_label = "Medium"
        else:
            risk_label = "High"

        # Save prediction
        import json

        prediction = Prediction(
            user_id=current_user.id,
            disease_type=disease,
            risk_probability=prob,
            risk_label=risk_label,
            input_data=json.dumps(input_data),
        )
        db.session.add(prediction)
        db.session.commit()

        return render_template(
            "result.html",
            disease_info=disease_info,
            probability=prob,
            risk_label=risk_label,
            input_data=input_data,
            feature_labels=FEATURE_LABELS,
        )

    return render_template(
        "predict.html",
        disease=disease,
        disease_info=disease_info,
        features=features,
        feature_labels=FEATURE_LABELS,
        dropdown_options=DROPDOWN_OPTIONS,
    )


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


# Create tables
with app.app_context():
    db.create_all()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
