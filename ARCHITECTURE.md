# MediSense AI Architecture

MediSense AI is organized as a Flask-first machine learning web application. The user-facing app lives in `webapp/`, while reusable data, model, training, and optional API code lives in `src/`.

## Runtime Architecture

```text
Browser
  |
  v
Flask Web App (webapp/app.py)
  |
  +-- Flask-Login user sessions
  +-- Flask-SQLAlchemy SQLite database
  +-- HTML templates in webapp/templates
  +-- CSS in webapp/static
  |
  v
Model Artifacts (models_saved/)
  |
  +-- preprocessor.pkl
  +-- stacking_ensemble.pkl
  |
  v
Core ML Code (src/)
```

The Flask app performs predictions directly. It loads the saved preprocessor and stacking ensemble from `models_saved/<disease>/`, transforms the submitted form data, calculates risk probability, saves the result to SQLite, and renders the result page.

## Main Components

| Component | Path | Responsibility |
| --- | --- | --- |
| Flask app | `webapp/app.py` | Routes, auth, database models, prediction flow |
| Templates | `webapp/templates/` | Landing, auth, dashboard, prediction, result, history pages |
| Styles | `webapp/static/css/style.css` | Visual styling for the Flask UI |
| Configuration | `src/config.py` | Dataset features, paths, model settings |
| Data pipeline | `src/data/` | Loading, preprocessing, splitting |
| Model code | `src/models/` | Base learners, deep learning model, stacking ensemble |
| Training | `src/training/` and `scripts/train_all.py` | Model training and artifact generation |
| Evaluation | `src/evaluation/` | Metrics and explainability helpers |
| Optional API | `src/api/` | FastAPI endpoints for separate API usage |
| Tests | `tests/` | API, ML, and Flask smoke tests |

## Flask Request Flow

```text
GET / or /dashboard
  -> Render Flask template

POST /signup or /login
  -> Validate form
  -> Create or authenticate user
  -> Store session with Flask-Login

GET /predict/<disease>
  -> Read feature list from src.config.DATASETS
  -> Render disease-specific input form

POST /predict/<disease>
  -> Parse submitted values
  -> Load model artifacts from models_saved/<disease>/
  -> Transform inputs with preprocessor.pkl
  -> Predict probability with stacking_ensemble.pkl
  -> Save Prediction row in SQLite
  -> Render result.html

GET /history
  -> Load current user's past predictions
  -> Render history.html
```

## Data and Model Flow

```text
data/raw/*.csv
  -> src/data/preprocessor.py
  -> src/models/stacking.py
  -> scripts/train_all.py
  -> models_saved/<disease>/
       +-- preprocessor.pkl
       +-- stacking_ensemble.pkl
  -> webapp/app.py prediction routes
```

## Deployment

The Flask web app can run locally:

```bash
python webapp/app.py
```

It can also run through Docker Compose:

```bash
cd docker
docker-compose up --build
```

Docker Compose starts the Flask web app on port `5000`.

## Optional API

`src/api/main.py` remains available for API tests or external integrations, but it is not required for the Flask app to work.

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```
