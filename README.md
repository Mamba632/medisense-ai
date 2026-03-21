# MediSense AI

**Intelligent Health Risk Prediction System**

MediSense AI is a machine-learning-powered platform that predicts disease risk for three major health conditions: heart disease, diabetes, and liver disease. It combines a stacking ensemble of classical ML models with an optional deep learning component, served through a FastAPI backend and an interactive Streamlit dashboard.

---

## Features

- **Multi-Disease Prediction** -- Heart disease, diabetes, and liver disease risk assessment from patient data.
- **Stacking Ensemble** -- Combines Random Forest, XGBoost, LightGBM, SVM, and KNN with a logistic regression meta-learner.
- **Optional Deep Learning** -- A TensorFlow neural network can be included as an additional base learner.
- **Automated Preprocessing** -- Outlier clipping, median imputation, standard scaling, and one-hot encoding per dataset.
- **Explainability** -- SHAP-based feature importance for model transparency.
- **REST API** -- FastAPI endpoints for predictions, health checks, and feature schema inspection.
- **Interactive Dashboard** -- Streamlit app with EDA explorer, risk predictor, model performance charts, and patient insights.
- **Dockerized Deployment** -- Docker Compose setup for one-command launch of API and dashboard.

---

## Tech Stack

| Layer | Technologies |
|-------|-------------|
| Machine Learning | scikit-learn, XGBoost, LightGBM, TensorFlow, imbalanced-learn |
| Explainability | SHAP |
| API | FastAPI, Uvicorn, Pydantic |
| Dashboard | Streamlit, Plotly, Matplotlib, Seaborn |
| Testing | pytest, httpx |
| Deployment | Docker, Docker Compose |
| Language | Python 3.11+ |

---

## Quick Start

```bash
# Clone the repository
git clone https://github.com/kuntardivyang/medisense-ai.git
cd medisense-ai

# Create a virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train all models
python scripts/train_all.py --dataset all

# Start the API
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# In another terminal, launch the dashboard
streamlit run app/streamlit_app.py --server.port 8501
```

For detailed setup instructions, Docker deployment, and troubleshooting, see [SETUP.md](SETUP.md).

---

## Project Structure

```
medisense-ai/
├── src/                        # Core source code
│   ├── config.py               # Settings, paths, hyperparameters
│   ├── data/                   # Data loading, preprocessing, splitting
│   │   └── preprocessor.py     # MediSensePreprocessor pipeline
│   ├── features/               # Feature engineering utilities
│   ├── models/                 # Base learners, DL model, stacking ensemble
│   │   └── stacking.py         # StackingEnsemble class
│   ├── evaluation/             # Metrics computation, SHAP explainability
│   │   └── metrics.py          # compute_metrics, format_metrics
│   ├── training/               # Training pipeline orchestration
│   └── api/                    # FastAPI REST endpoints
│       ├── main.py             # App entry point and routes
│       ├── health.py           # /health endpoint
│       ├── predict.py          # Prediction logic
│       └── schemas.py          # Pydantic input/output models
├── app/                        # Streamlit dashboard
│   ├── streamlit_app.py        # Main entry point
│   ├── pages/                  # EDA, Predictor, Performance, Insights
│   └── components/             # Reusable chart and sidebar components
├── data/
│   ├── raw/                    # Original datasets (CSV)
│   ├── processed/              # Cleaned/engineered data
│   └── splits/                 # Train/val/test splits
├── models_saved/               # Trained model artifacts (.pkl)
├── notebooks/                  # Jupyter EDA notebooks
├── scripts/                    # CLI training scripts
├── docker/                     # Dockerfiles and docker-compose.yml
├── tests/                      # Unit tests
│   ├── test_preprocessor.py
│   ├── test_stacking.py
│   ├── test_api.py
│   └── test_metrics.py
├── requirements.txt            # Python dependencies
├── SETUP.md                    # Detailed setup and run guide
└── ARCHITECTURE.md             # System architecture documentation
```

---

## Model Results

Performance on held-out test sets using the stacking ensemble:

| Disease | ROC-AUC | Description |
|---------|---------|-------------|
| Heart Disease | **0.9390** | Cleveland Heart Disease dataset (303 samples) |
| Diabetes | **0.8367** | PIMA Indians Diabetes dataset (768 samples) |
| Liver Disease | **0.7365** | Indian Liver Patient dataset (583 samples) |

---

## Screenshots

> Screenshots of the Streamlit dashboard will be added here.

<!--
![EDA Explorer](docs/screenshots/eda_explorer.png)
![Risk Predictor](docs/screenshots/risk_predictor.png)
![Model Performance](docs/screenshots/model_performance.png)
![Patient Insights](docs/screenshots/patient_insights.png)
-->

---

## Author

**Raval Manav**

Supervisor: **Prof. Darshana Patel**

---

## License

This project is licensed under the [MIT License](LICENSE).
