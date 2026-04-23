# MediSense AI Setup and Run Guide

This project now uses Flask as the main web application. The old dashboard has been removed, and the app should be run as the `webapp.app` module.

## Prerequisites

- Python 3.11 or 3.12
- pip
- Git
- Docker and Docker Compose, optional

## 1. Create a Virtual Environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

Linux or macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

## 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```

Installed groups include:

- ML: scikit-learn, XGBoost, LightGBM, TensorFlow, SHAP, imbalanced-learn
- Web app: Flask, Flask-Login, Flask-SQLAlchemy
- Optional API: FastAPI, Uvicorn, Pydantic
- Testing: pytest, httpx, ruff

## 3. Verify Datasets

The datasets should exist in `data/raw/`:

```bash
dir data\raw
```

Expected files:

```text
heart.csv
diabetes.csv
indian_liver_patient.csv
README.md
```

## 4. Train Models

Train all disease models:

```bash
python -m scripts.train_all --dataset all
```

Train one model:

```bash
python -m scripts.train_all --dataset heart
python -m scripts.train_all --dataset diabetes
python -m scripts.train_all --dataset liver
```

Faster training without the deep learning learner:

```bash
python -m scripts.train_all --dataset all --no-dl
```

Model artifacts are saved in:

```text
models_saved/<dataset>/
```

## 5. Run the Flask App

```bash
python webapp/app.py
```

On Windows, you can also run:

```powershell
.\run_webapp.ps1
```

Open:

```text
http://localhost:5000
```

The app creates the local SQLite database automatically. For local development, it also uses a development-only session secret when `SECRET_KEY` is not set.

The Flask app includes:

- Landing page
- Signup and login
- User dashboard
- Disease prediction forms
- AI risk result page
- Prediction history

For production-style database changes, use Flask-Migrate commands after installing dependencies:

```bash
flask --app webapp.app db init
flask --app webapp.app db migrate -m "describe change"
flask --app webapp.app db upgrade
```

## 6. Environment Variables

You can copy `.env.example` to `.env` if you want local configuration:

```text
FLASK_HOST=127.0.0.1
FLASK_PORT=5000
FLASK_DEBUG=0
SECRET_KEY=
MEDISENSE_ENV=development
```

`SECRET_KEY` is optional for local development, but production deployments should set a long random value and use `MEDISENSE_ENV=production`.

## 7. Optional FastAPI Server

The Flask web app works without running FastAPI. To test the API separately:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Available endpoints:

```text
http://localhost:8000/docs
http://localhost:8000/health
http://localhost:8000/models
```

## 8. Run Tests

```bash
pytest tests/ -v
```

## 9. Run with Docker

```bash
cd docker
docker-compose up --build
```

Docker Compose also requires a real `SECRET_KEY`; set it in `docker/.env` or in your shell before running.

Open:

```text
http://localhost:5000
```

## Troubleshooting

| Issue | Solution |
| --- | --- |
| `ModuleNotFoundError` for ML packages | Run `pip install -r requirements.txt` |
| `RuntimeError` for `SECRET_KEY` | Set a unique `SECRET_KEY` in `.env` or your shell |
| `FileNotFoundError` for model artifacts | Run `python -m scripts.train_all --dataset all` |
| Flask page opens but prediction fails | Check that `models_saved/<dataset>/preprocessor.pkl` and `stacking_ensemble.pkl` exist |
| Port already in use | Set `FLASK_PORT` to another port before running |
| Docker build is slow | TensorFlow and ML packages are large; this is expected on first build |
