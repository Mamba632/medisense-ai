# MediSense AI — Setup & Run Guide

## Prerequisites

- Python 3.11 or 3.12
- pip (Python package manager)
- Git
- Docker & Docker Compose (optional, for containerized deployment)

---

## Step 1: Clone the Repository

```bash
git clone https://github.com/kuntardivyang/medisense-ai.git
cd medisense-ai
```

---

## Step 2: Create a Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows
```

---

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required libraries:
- **ML**: scikit-learn, XGBoost, LightGBM, TensorFlow, SHAP, imbalanced-learn
- **Visualization**: matplotlib, seaborn, plotly
- **API**: FastAPI, uvicorn, pydantic
- **Dashboard**: Streamlit
- **Testing**: pytest, httpx

---

## Step 4: Verify Datasets

The datasets are included in `data/raw/`. Verify they exist:

```bash
ls data/raw/
# Should show: heart.csv  diabetes.csv  indian_liver_patient.csv  README.md
```

| Dataset | Rows | Description |
|---------|------|-------------|
| heart.csv | 303 | Heart Disease (Cleveland) |
| diabetes.csv | 768 | PIMA Indians Diabetes |
| indian_liver_patient.csv | 583 | Indian Liver Patient |

---

## Step 5: Train the Models

Train all 3 disease models with stacking ensemble (includes deep learning):

```bash
python scripts/train_all.py --dataset all
```

Or train a specific model:

```bash
python scripts/train_all.py --dataset heart
python scripts/train_all.py --dataset diabetes
python scripts/train_all.py --dataset liver
```

To skip deep learning (faster, no TensorFlow needed):

```bash
python scripts/train_all.py --dataset all --no-dl
```

Trained models are saved to `models_saved/<dataset>/`.

---

## Step 6: Run the FastAPI Server

Start the prediction API:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at:
- Swagger docs: http://localhost:8000/docs
- Health check: http://localhost:8000/health
- List models: http://localhost:8000/models

### Test a prediction:

```bash
curl -X POST http://localhost:8000/predict/heart \
  -H "Content-Type: application/json" \
  -d '{
    "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
    "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
    "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
  }'
```

---

## Step 7: Run the Streamlit Dashboard

In a separate terminal (keep the API running):

```bash
streamlit run app/streamlit_app.py --server.port 8501
```

Open http://localhost:8501 in your browser. The dashboard has 4 pages:

1. **EDA Explorer** — Interactive data exploration with charts
2. **Risk Predictor** — Enter patient data, get risk assessment (requires API running)
3. **Model Performance** — ROC curves, AUC comparison across models
4. **Patient Insights** — Population-level health analytics

---

## Step 8: Run Tests

```bash
pytest tests/ -v
```

---

## Alternative: Run with Docker

Build and start both services at once:

```bash
cd docker
docker-compose up --build
```

This starts:
- FastAPI at http://localhost:8000
- Streamlit at http://localhost:8501

---

## Project Structure Overview

```
medisense-ai/
├── src/                    # Core source code
│   ├── config.py           # All settings, paths, hyperparameters
│   ├── data/               # Data loading, preprocessing, splitting
│   ├── features/           # Feature engineering
│   ├── models/             # Base learners, DL model, stacking ensemble
│   ├── evaluation/         # Metrics, SHAP explainability
│   ├── training/           # Training pipeline
│   └── api/                # FastAPI REST endpoints
├── app/                    # Streamlit dashboard
│   ├── streamlit_app.py    # Main entry point
│   ├── pages/              # 4 dashboard pages
│   └── components/         # Reusable charts and sidebar
├── data/raw/               # Original datasets (CSV)
├── models_saved/           # Trained model artifacts (.pkl)
├── notebooks/              # Jupyter EDA notebooks
├── scripts/                # CLI training script
├── docker/                 # Dockerfiles and docker-compose
├── tests/                  # Unit tests
└── requirements.txt        # Python dependencies
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'tensorflow'` | Run `pip install tensorflow` or use `--no-dl` flag |
| `FileNotFoundError: Dataset file not found` | Ensure `data/raw/` contains the 3 CSV files |
| `ConnectionError` on Risk Predictor page | Start the FastAPI server first (Step 6) |
| Models not found on Performance page | Train models first (Step 5) |
| Docker build fails | Ensure Docker daemon is running and you have sufficient disk space |
