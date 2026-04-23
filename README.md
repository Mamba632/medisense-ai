# MediSense AI

Intelligent Health Risk Prediction System

MediSense AI is a machine-learning-powered Flask web application that predicts disease risk for heart disease, diabetes, and liver disease. It includes user signup/login, a patient dashboard, prediction history, and trained ML model loading from the Flask app.

The project keeps the core ML package in `src/` and serves the user-facing experience through `webapp/app.py`.

## Features

- Multi-disease prediction for heart disease, diabetes, and liver disease.
- Flask web app with signup, login, dashboard, prediction forms, results, and history.
- SQLite database through Flask-SQLAlchemy.
- Stacking ensemble models saved under `models_saved/`.
- Automated preprocessing with the saved preprocessor artifacts.
- Optional FastAPI prediction API remains in `src/api/` for API testing or future integrations.
- Docker support for running the Flask web app.

## Tech Stack

| Layer | Technologies |
| --- | --- |
| Web App | Flask, Flask-Login, Flask-SQLAlchemy, Bootstrap |
| Machine Learning | scikit-learn, XGBoost, LightGBM, TensorFlow, imbalanced-learn |
| Data and Explainability | pandas, numpy, SHAP |
| Optional API | FastAPI, Uvicorn, Pydantic |
| Testing | pytest, httpx |
| Deployment | Docker, Docker Compose |

## Quick Start

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python webapp/app.py
```

Open the Flask app at:

```text
http://localhost:5000
```

The local app creates its SQLite database automatically and uses a development-only session secret if `SECRET_KEY` is not set.

If trained model files are missing from `models_saved/`, train them once:

```bash
python -m scripts.train_all --dataset all
```

Windows shortcut after setup, if you prefer PowerShell:

```powershell
.\run_webapp.ps1
```

## Optional FastAPI Server

The Flask app does not need FastAPI to run. If you want to test the API separately:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

API docs:

```text
http://localhost:8000/docs
```

## Docker

```bash
cd docker
docker-compose up --build
```

The Flask app will be available at:

```text
http://localhost:5000
```

## Project Structure

```text
medisense-ai/
+-- src/                    # Core ML, data, training, evaluation, optional API
+-- webapp/                 # Flask web application
|   +-- app.py              # Flask entry point
|   +-- templates/          # HTML templates
|   +-- static/             # CSS and static assets
+-- data/raw/               # Original datasets
+-- models_saved/           # Trained model artifacts
+-- notebooks/              # EDA notebooks
+-- scripts/                # Training scripts
+-- docker/                 # Docker files
+-- tests/                  # Unit and smoke tests
+-- requirements.txt
+-- SETUP.md
+-- ARCHITECTURE.md
```

## Run Tests

```bash
pip install -r requirements-dev.txt
pip install -e .
pytest tests/ -v
```

## Author

Raval Manav

Supervisor: Prof. Darshana Patel
