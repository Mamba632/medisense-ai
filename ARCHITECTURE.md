# MediSense AI — Intelligent Health Risk Prediction System
## Full Architecture Document

**Author**: Raval Manav
**Supervisor**: Prof. Darshana Patel
**Date**: 2026-03-21

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Folder Structure](#2-folder-structure)
3. [Dataset Recommendations](#3-dataset-recommendations)
4. [Tech Stack](#4-tech-stack)
5. [ML Pipeline Architecture](#5-ml-pipeline-architecture)
6. [Deep Learning Component](#6-deep-learning-component)
7. [Stacking Ensemble Design](#7-stacking-ensemble-design)
8. [Deployment Architecture](#8-deployment-architecture)
9. [Visualization Strategy](#9-visualization-strategy)
10. [Build Sequence (Ordered Phases)](#10-build-sequence-ordered-phases)
11. [Risk Assessment](#11-risk-assessment)
12. [Implementation Plan (Task List)](#12-implementation-plan-task-list)

---

## 1. Project Overview

MediSense AI is a cloud-ready, ML-powered health risk prediction system. Given a patient's clinical indicators (e.g., blood pressure, glucose, cholesterol, BMI), it predicts the risk of one or more diseases using a Stacking Ensemble of multiple classical ML models and a deep learning base learner. Results are exposed via a REST API and an interactive Streamlit dashboard.

### Core Goals

| Goal | Mechanism |
|------|-----------|
| Uncover patterns in health data | Comprehensive EDA notebooks |
| Accurate risk prediction | Stacking Ensemble (ML + DL) |
| Interpretability | SHAP explanations per prediction |
| Interactive insights | Streamlit dashboard + Plotly charts |
| Scalable deployment | FastAPI + Docker + cloud (Render/GCP/AWS) |

---

## 2. Folder Structure

```
medisense-ai/
│
├── data/
│   ├── raw/                    # Original downloaded CSVs (never modified)
│   │   ├── heart_disease.csv
│   │   ├── diabetes.csv
│   │   ├── liver_disease.csv
│   │   └── README.md           # Dataset sources and licenses
│   ├── processed/              # Cleaned, feature-engineered data
│   │   ├── heart_disease_processed.csv
│   │   ├── diabetes_processed.csv
│   │   └── liver_disease_processed.csv
│   └── splits/                 # Train/val/test splits (reproducible)
│       ├── heart/
│       ├── diabetes/
│       └── liver/
│
├── notebooks/                  # Jupyter notebooks (EDA + experiments)
│   ├── 01_eda_heart_disease.ipynb
│   ├── 02_eda_diabetes.ipynb
│   ├── 03_eda_liver_disease.ipynb
│   ├── 04_preprocessing_pipeline.ipynb
│   ├── 05_baseline_models.ipynb
│   ├── 06_deep_learning_model.ipynb
│   ├── 07_stacking_ensemble.ipynb
│   └── 08_model_evaluation_shap.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py               # Global constants: paths, hyperparams, seeds
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py           # load_raw_dataset(name) -> pd.DataFrame
│   │   ├── preprocessor.py     # MediSensePreprocessor (sklearn Pipeline)
│   │   └── splitter.py         # stratified_split(df, target, seed) -> splits
│   │
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineering.py      # add_risk_score(), encode_categoricals(), etc.
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_learners.py    # get_base_learners() -> list[sklearn estimators]
│   │   ├── dl_model.py         # build_dl_model(input_dim) -> keras.Model
│   │   ├── stacking.py         # StackingEnsemble class
│   │   └── meta_learner.py     # LogisticRegression meta-learner wrapper
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py          # compute_metrics(y_true, y_pred, y_prob)
│   │   └── explainability.py   # shap_summary(model, X) -> shap.Explanation
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py          # train_pipeline(dataset_name) entry point
│   │
│   └── api/
│       ├── __init__.py
│       ├── main.py             # FastAPI app factory
│       ├── schemas.py          # Pydantic request/response models
│       ├── predict.py          # /predict endpoint logic
│       └── health.py           # /health endpoint
│
├── app/
│   ├── streamlit_app.py        # Main Streamlit dashboard entry point
│   ├── pages/
│   │   ├── 01_eda_explorer.py
│   │   ├── 02_risk_predictor.py
│   │   ├── 03_model_performance.py
│   │   └── 04_patient_insights.py
│   └── components/
│       ├── charts.py           # Reusable Plotly chart functions
│       └── sidebar.py          # Sidebar inputs and controls
│
├── models_saved/               # Serialized model artifacts
│   ├── heart/
│   │   ├── base_learners/
│   │   │   ├── rf_model.pkl
│   │   │   ├── xgb_model.pkl
│   │   │   ├── lgbm_model.pkl
│   │   │   ├── svm_model.pkl
│   │   │   └── knn_model.pkl
│   │   ├── dl_model/
│   │   │   ├── model.keras
│   │   │   └── scaler.pkl
│   │   ├── meta_learner.pkl
│   │   ├── stacking_ensemble.pkl
│   │   └── preprocessor.pkl
│   ├── diabetes/
│   └── liver/
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessor.py
│   ├── test_stacking.py
│   ├── test_api.py
│   └── test_metrics.py
│
├── docker/
│   ├── Dockerfile.api          # FastAPI container
│   ├── Dockerfile.streamlit    # Streamlit container
│   └── docker-compose.yml      # Local multi-service orchestration
│
├── scripts/
│   ├── download_datasets.sh    # kaggle CLI download commands
│   ├── train_all.py            # CLI: python scripts/train_all.py --dataset heart
│   └── evaluate_all.py         # CLI: evaluate all trained models
│
├── requirements.txt            # Python dependencies (pinned versions)
├── requirements-dev.txt        # Dev/test dependencies
├── .env.example                # Environment variable template
├── .gitignore
├── README.md
└── ARCHITECTURE.md             # This document
```

---

## 3. Dataset Recommendations

All datasets are publicly available under open licenses with no ethical barriers for academic use.

| Dataset | Source | Target | Features | Rows | License |
|---------|--------|--------|----------|------|---------|
| Heart Disease (Cleveland) | UCI ML Repo / Kaggle | `target` (0/1) | age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal | 303 | CC BY 4.0 |
| PIMA Indians Diabetes | Kaggle / UCI | `Outcome` (0/1) | Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age | 768 | Public Domain |
| Indian Liver Patient Dataset | UCI / Kaggle | `Dataset` (1/2) | Age, Gender, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio | 583 | CC BY 4.0 |

### Download Instructions

```bash
# Install Kaggle CLI
pip install kaggle

# Download datasets (requires ~/.kaggle/kaggle.json API key)
kaggle datasets download -d ronitf/heart-disease-uci -p data/raw/
kaggle datasets download -d uciml/pima-indians-diabetes-database -p data/raw/
kaggle datasets download -d uciml/indian-liver-patient-records -p data/raw/
```

### Why These Datasets

- Well-studied benchmarks with published baseline results for comparison
- Binary classification targets (simplifies stacking meta-learner)
- Tabular format (ideal for classical ML + tabular DL)
- No privacy concerns: anonymized, no HIPAA-protected fields

---

## 4. Tech Stack

### Core ML / Data

| Library | Version | Purpose |
|---------|---------|---------|
| `pandas` | 2.2.x | Tabular data manipulation |
| `numpy` | 1.26.x | Numerical operations |
| `scikit-learn` | 1.4.x | Base learners, preprocessing, Pipeline, metrics |
| `xgboost` | 2.0.x | Gradient boosted base learner |
| `lightgbm` | 4.3.x | Fast gradient boosting base learner |
| `tensorflow` / `keras` | 2.15.x | Deep learning base learner (MLP) |
| `imbalanced-learn` | 0.12.x | SMOTE for class imbalance |
| `shap` | 0.44.x | Model explainability |
| `optuna` | 3.5.x | Hyperparameter tuning |

### EDA / Visualization

| Library | Version | Purpose |
|---------|---------|---------|
| `matplotlib` | 3.8.x | Static plots in notebooks |
| `seaborn` | 0.13.x | Statistical EDA plots |
| `plotly` | 5.19.x | Interactive charts in Streamlit |
| `kaleido` | 0.2.x | Plotly static export |

### API / Deployment

| Library | Version | Purpose |
|---------|---------|---------|
| `fastapi` | 0.110.x | REST API server |
| `uvicorn` | 0.29.x | ASGI server |
| `pydantic` | 2.6.x | Request/response validation |
| `streamlit` | 1.32.x | Interactive dashboard |
| `joblib` | 1.3.x | Model serialization |
| `python-dotenv` | 1.0.x | Environment variable management |

### Infrastructure

| Tool | Purpose |
|------|---------|
| Docker + Docker Compose | Local containerized deployment |
| Render / Railway (primary) | Free-tier cloud deployment for student project |
| GitHub Actions | CI: test + lint on push |
| pytest | Unit and integration tests |

### Rationale: Why FastAPI + Streamlit (not Flask)

- FastAPI: async-native, automatic OpenAPI docs, Pydantic validation — better for ML APIs that may add async preprocessing later
- Streamlit: minimal boilerplate, Python-only, renders Plotly natively — ideal for ML demo dashboards
- They run as separate services (decoupled), communicating via HTTP — clean architecture

---

## 5. ML Pipeline Architecture

```
Raw CSV
   │
   ▼
[DataLoader]          src/data/loader.py
   │  load_raw_dataset(name)
   ▼
[Preprocessing Pipeline]  src/data/preprocessor.py
   │  Steps (sklearn.Pipeline):
   │  1. ColumnSelector      — drop irrelevant columns
   │  2. MissingValueImputer — median for numeric, mode for categorical
   │  3. OutlierClipper      — IQR-based clipping (1.5x)
   │  4. FeatureEngineer     — custom transforms (see §5.1)
   │  5. StandardScaler      — numeric features
   │  6. OneHotEncoder       — categorical features
   ▼
[Stratified Split]    src/data/splitter.py
   │  Train 70% | Val 15% | Test 15%  (seed=42, stratify=target)
   ▼
[SMOTE Oversampling]  (train set only — never on val/test)
   │
   ▼
[Base Learner Training]  src/models/base_learners.py
   │  Each trained independently with cross-validation:
   │  - RandomForestClassifier     (n_estimators=200)
   │  - XGBClassifier              (tree_method=hist)
   │  - LGBMClassifier             (boosting_type=gbdt)
   │  - SVC(probability=True)      (kernel=rbf)
   │  - KNeighborsClassifier       (n_neighbors=tuned)
   │  - MLPClassifier (sklearn)    (baseline tabular DL, replaced by Keras)
   │
   ▼
[Deep Learning Base Learner]  src/models/dl_model.py
   │  Keras MLP (see §6)
   ▼
[Stacking Ensemble]   src/models/stacking.py
   │  Meta-features: OOF predictions from base learners
   │  Meta-learner: LogisticRegression (C=tuned)
   ▼
[Evaluation]          src/evaluation/metrics.py
   │  Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC
   │  Confusion matrix, calibration curve
   ▼
[Explainability]      src/evaluation/explainability.py
      SHAP TreeExplainer / KernelExplainer → feature importance
```

### 5.1 Feature Engineering Details

| Transform | Logic | Datasets |
|-----------|-------|----------|
| `bmi_category` | Underweight/Normal/Overweight/Obese buckets from BMI | Diabetes, Liver |
| `age_group` | Child/Young/Middle/Senior bins from Age | All |
| `glucose_insulin_ratio` | Glucose / (Insulin + 1) — insulin resistance proxy | Diabetes |
| `cardiac_risk_score` | Composite: trestbps + chol + (thalach_inv) normalized | Heart |
| `liver_enzyme_ratio` | Alamine_AT / Aspartate_AT ratio | Liver |

---

## 6. Deep Learning Component

### Architecture: Tabular MLP (Multi-Layer Perceptron)

File: `src/models/dl_model.py`

```python
def build_dl_model(input_dim: int, dropout_rate: float = 0.3) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(256, activation="relu")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout_rate)(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dropout(dropout_rate / 2)(x)
    output = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, output)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")]
    )
    return model
```

### Training Strategy

- **Callbacks**: EarlyStopping(patience=10, monitor=val_auc), ReduceLROnPlateau
- **Epochs**: max 200 (early stopping expected to fire at ~30-80)
- **Batch size**: 32
- **Validation**: 15% of training set held out as Keras validation
- **Output**: probability score (sigmoid) — fed into stacking as one column of meta-features

### Integration with Stacking

The Keras model is wrapped in a scikit-learn-compatible `KerasClassifierWrapper` (implements `fit`, `predict_proba`) so it participates in the stacking pipeline uniformly with sklearn estimators.

```python
class KerasClassifierWrapper(BaseEstimator, ClassifierMixin):
    def fit(self, X, y): ...          # trains Keras model
    def predict_proba(self, X): ...   # returns [P(0), P(1)]
    def predict(self, X): ...         # thresholds at 0.5
```

---

## 7. Stacking Ensemble Design

### Two-Layer Stack

```
Layer 1 — Base Learners (6 models):
  RF | XGBoost | LightGBM | SVM | KNN | Keras-MLP
         │           │           │        │        │       │
         └───────────┴───────────┴────────┴────────┘
                              │
                  Out-of-Fold (OOF) predictions
                  (5-fold stratified CV on train set)
                              │
                              ▼
Layer 2 — Meta-Learner:
  LogisticRegression trained on [OOF_RF, OOF_XGB, OOF_LGBM,
                                  OOF_SVM, OOF_KNN, OOF_Keras]
                              │
                              ▼
                    Final Risk Probability
```

### Key Implementation Details

**File**: `src/models/stacking.py`

```python
class StackingEnsemble:
    def __init__(self, base_learners: list, meta_learner, n_folds: int = 5):
        ...

    def fit(self, X_train, y_train) -> "StackingEnsemble":
        # 1. Generate OOF predictions via cross_val_predict(method="predict_proba")
        # 2. Stack OOF columns into meta-feature matrix S_train
        # 3. Retrain each base learner on full X_train
        # 4. Train meta_learner on S_train, y_train
        ...

    def predict_proba(self, X_test) -> np.ndarray:
        # 1. Get predictions from each fully-trained base learner
        # 2. Stack into S_test
        # 3. Return meta_learner.predict_proba(S_test)
        ...
```

### Why Stacking Over Simple Voting

- Different models capture different patterns (e.g., XGBoost handles non-linearity, SVM captures margin boundaries)
- OOF training prevents meta-learner from overfitting to base learner outputs
- Meta-learner learns optimal weights per base learner on validation evidence
- Empirically outperforms Voting and Bagging on tabular medical data

---

## 8. Deployment Architecture

### Local Development

```
docker-compose up
  ├── fastapi service   → localhost:8000  (src/api/main.py)
  └── streamlit service → localhost:8501  (app/streamlit_app.py)
```

### Cloud Deployment (Render — free tier suitable for student project)

```
GitHub Repository
       │
       ▼  (push to main triggers)
GitHub Actions CI
  - pytest
  - lint (ruff)
       │
       ▼  (on success)
Render Auto-Deploy
  ├── Web Service 1: FastAPI (Docker, src/api/)
  │     URL: https://medisense-api.onrender.com
  └── Web Service 2: Streamlit (Docker, app/)
        URL: https://medisense-app.onrender.com
```

### FastAPI Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| GET | `/models` | List available disease models |
| POST | `/predict/{disease}` | Predict risk for given disease |
| GET | `/predict/{disease}/features` | Return expected feature schema |
| POST | `/predict/{disease}/batch` | Batch prediction (list of patients) |

### Pydantic Schemas (example for heart disease)

```python
class HeartDiseaseInput(BaseModel):
    age: int = Field(..., ge=1, le=120)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)          # chest pain type
    trestbps: float = Field(..., ge=60, le=250)
    chol: float = Field(..., ge=100, le=600)
    fbs: int = Field(..., ge=0, le=1)
    # ... all 13 features with validation ranges

class RiskPredictionOutput(BaseModel):
    disease: str
    risk_probability: float
    risk_label: str                            # "Low" / "Medium" / "High"
    risk_threshold: float
    top_risk_factors: list[dict]               # SHAP-derived
    model_version: str
```

### Dockerfiles

**`docker/Dockerfile.api`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ ./src/
COPY models_saved/ ./models_saved/
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker/Dockerfile.streamlit`**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app/ ./app/
COPY data/processed/ ./data/processed/
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## 9. Visualization Strategy

### Notebook EDA Visualizations (seaborn/matplotlib)

| Chart | Purpose | Library |
|-------|---------|---------|
| Correlation heatmap | Feature-to-feature and feature-to-target correlations | seaborn |
| Distribution plots (KDE) | Per-feature distribution split by target class | seaborn |
| Pair plot | Multi-feature scatterplot matrix | seaborn |
| Box plots | Outlier detection per feature | matplotlib |
| Missing value heatmap | Visualize missingness patterns | seaborn/missingno |
| Class balance bar chart | Target imbalance before/after SMOTE | matplotlib |

### Streamlit Dashboard Pages

**Page 1 — EDA Explorer** (`app/pages/01_eda_explorer.py`)
- Dataset selector dropdown (Heart / Diabetes / Liver)
- Interactive Plotly histogram with KDE per feature
- Correlation heatmap (Plotly heatmap)
- Scatter matrix with color by class

**Page 2 — Risk Predictor** (`app/pages/02_risk_predictor.py`)
- Patient input form (sliders and dropdowns matching feature ranges)
- Submit button → POST to FastAPI `/predict/{disease}`
- Risk gauge chart (Plotly indicator)
- SHAP waterfall chart for this patient's top contributing features

**Page 3 — Model Performance** (`app/pages/03_model_performance.py`)
- ROC curve (all base learners + stacking ensemble overlaid)
- Precision-Recall curve
- Confusion matrix heatmap
- Feature importance bar chart (aggregated SHAP)
- Model comparison table (Accuracy / F1 / AUC per model)

**Page 4 — Patient Insights** (`app/pages/04_patient_insights.py`)
- Population-level SHAP summary beeswarm plot
- Risk distribution histogram across dataset
- Demographic risk breakdown (age group vs. disease risk)

### Key Visualization Decisions

- Use Plotly throughout Streamlit (not matplotlib) for interactivity and zoom/hover
- SHAP `waterfall_plot` and `summary_plot` saved as static images during training and loaded in dashboard (avoids SHAP dependency in the frontend container)
- All charts wrapped in reusable functions in `app/components/charts.py`

---

## 10. Build Sequence (Ordered Phases)

### Phase 0 — Project Bootstrap (Day 1)
1. Initialize repo, create folder structure, `.gitignore`, `requirements.txt`
2. Download datasets via Kaggle CLI, place in `data/raw/`
3. Write `src/config.py` with all dataset paths, random seeds, column names

### Phase 1 — EDA (Days 2-4)
4. `notebooks/01_eda_heart_disease.ipynb` — full EDA with all charts
5. `notebooks/02_eda_diabetes.ipynb`
6. `notebooks/03_eda_liver_disease.ipynb`
7. Document findings in `data/raw/README.md` (class imbalance, missing value rates, key correlations)

### Phase 2 — Data Pipeline (Days 5-7)
8. Implement `src/data/loader.py` with `load_raw_dataset(name)`
9. Implement `src/data/preprocessor.py` (sklearn Pipeline: imputation, outlier clip, scaling, encoding)
10. Implement `src/features/engineering.py` (dataset-specific feature transforms)
11. Implement `src/data/splitter.py` with reproducible stratified split
12. `notebooks/04_preprocessing_pipeline.ipynb` — validate pipeline visually

### Phase 3 — Model Development (Days 8-14)
13. Implement `src/models/base_learners.py` — define and train 5 sklearn base learners
14. `notebooks/05_baseline_models.ipynb` — baseline evaluation with CV
15. Implement `src/models/dl_model.py` — Keras MLP + `KerasClassifierWrapper`
16. `notebooks/06_deep_learning_model.ipynb` — DL training + evaluation
17. Implement `src/models/stacking.py` — `StackingEnsemble` class with OOF logic
18. Implement `src/models/meta_learner.py` — meta-learner wrapper
19. `notebooks/07_stacking_ensemble.ipynb` — train and evaluate full stack

### Phase 4 — Evaluation and Explainability (Days 15-17)
20. Implement `src/evaluation/metrics.py` — all metric functions
21. Implement `src/evaluation/explainability.py` — SHAP wrappers
22. `notebooks/08_model_evaluation_shap.ipynb` — full evaluation suite
23. Save all model artifacts to `models_saved/`
24. Implement `scripts/train_all.py` — single command to reproduce training

### Phase 5 — API Development (Days 18-20)
25. Implement `src/api/schemas.py` — Pydantic models for all 3 diseases
26. Implement `src/api/predict.py` — load model from disk, run inference
27. Implement `src/api/main.py` — FastAPI app with all routes
28. Implement `src/api/health.py` — `/health` endpoint
29. Write `tests/test_api.py` — pytest + httpx integration tests

### Phase 6 — Streamlit Dashboard (Days 21-24)
30. Implement `app/components/charts.py` — reusable Plotly chart functions
31. Implement `app/components/sidebar.py` — shared sidebar widget
32. Implement `app/pages/01_eda_explorer.py`
33. Implement `app/pages/02_risk_predictor.py` (calls FastAPI)
34. Implement `app/pages/03_model_performance.py`
35. Implement `app/pages/04_patient_insights.py`
36. Implement `app/streamlit_app.py` — main entry point + navigation

### Phase 7 — Docker and Deployment (Days 25-27)
37. Write `docker/Dockerfile.api` and `docker/Dockerfile.streamlit`
38. Write `docker/docker-compose.yml` — test local multi-service setup
39. Set up Render account + configure two Web Services
40. Add `.github/workflows/ci.yml` — pytest + ruff on push to main
41. Push to GitHub, verify Render auto-deploy succeeds

### Phase 8 — Tests, Docs, Polish (Days 28-30)
42. Write `tests/test_preprocessor.py` and `tests/test_stacking.py`
43. Write `tests/test_metrics.py`
44. Write `README.md` with setup, usage, deployment instructions
45. Final report writeup / presentation prep

---

## 11. Risk Assessment

| Risk | Category | Likelihood | Impact | Score | Mitigation |
|------|----------|-----------|--------|-------|------------|
| Small dataset size (303-768 rows) — overfitting on stacked model | TECHNICAL | 0.7 | 0.7 | 0.49 | SMOTE, extensive CV, strong regularization on meta-learner |
| Keras model underperforms on small tabular data | TECHNICAL | 0.6 | 0.4 | 0.24 | Treat DL as optional base learner; fallback to sklearn MLP if Keras adds no lift |
| Render free tier cold-start (30s+ spin-up) | EXTERNAL | 0.8 | 0.3 | 0.24 | Document expected latency; add loading spinner in Streamlit |
| SHAP calculation slow for SVM/Keras | TECHNICAL | 0.6 | 0.3 | 0.18 | Pre-compute SHAP values at training time; serve cached images |
| Streamlit ↔ FastAPI network latency on cloud | EXTERNAL | 0.4 | 0.3 | 0.12 | Use direct in-process call option if both deployed on same service |
| Class imbalance degrading recall | QUALITY | 0.6 | 0.5 | 0.30 | SMOTE + class_weight="balanced" + optimize for F1/AUC not accuracy |
| Dependency conflicts (TF + sklearn + joblib versions) | TECHNICAL | 0.4 | 0.4 | 0.16 | Pin all versions in requirements.txt; test in Docker first |

**Overall Risk Level**: MEDIUM (highest single score 0.49; average 0.25)

**Primary Mitigation Priority**: Overfitting on small datasets — use nested CV, enforce strong meta-learner regularization, report test metrics only after final model selection.

---

## 12. Implementation Plan (Task List)

### P0 — Foundation (blocks all other work)

- [ ] **T01**: Create full folder structure, `.gitignore`, `requirements.txt` with pinned versions -- `./` root -- Done: `git status` shows all dirs, `pip install -r requirements.txt` succeeds
- [ ] **T02**: Download and verify all 3 datasets -- `data/raw/` -- Done: 3 CSV files present, shapes match documented row counts
- [ ] **T03**: Implement `src/config.py` with dataset paths, column name lists, seeds -- `src/config.py` -- Done: importable, all constants accessible

### P0 — Data Pipeline (blocks model training)

- [ ] **T04**: Implement `src/data/loader.py` -- `src/data/loader.py` -- Done: `load_raw_dataset("heart")` returns correct DataFrame shape
- [ ] **T05**: Implement `src/data/preprocessor.py` (full sklearn Pipeline) -- `src/data/preprocessor.py` -- Done: `preprocessor.fit_transform(df)` produces clean numeric matrix, no NaNs
- [ ] **T06**: Implement `src/features/engineering.py` -- `src/features/engineering.py` -- Done: engineered features present in output DataFrame
- [ ] **T07**: Implement `src/data/splitter.py` -- `src/data/splitter.py` -- Done: train/val/test proportions match 70/15/15, stratification verified

### P1 — EDA Notebooks

- [ ] **T08**: `notebooks/01_eda_heart_disease.ipynb` -- `notebooks/` -- Done: all 6 EDA chart types rendered, key findings documented in markdown cells
- [ ] **T09**: `notebooks/02_eda_diabetes.ipynb` -- same criteria as T08
- [ ] **T10**: `notebooks/03_eda_liver_disease.ipynb` -- same criteria as T08

### P1 — Model Training

- [ ] **T11**: Implement `src/models/base_learners.py` (5 sklearn models) -- `src/models/base_learners.py` -- Done: `get_base_learners()` returns list of 5 fitted estimators with CV AUC > 0.70
- [ ] **T12**: Implement `src/models/dl_model.py` + `KerasClassifierWrapper` -- `src/models/dl_model.py` -- Done: `wrapper.fit(X_train, y_train)` runs; `wrapper.predict_proba(X_test)` returns shape (n, 2)
- [ ] **T13**: Implement `src/models/stacking.py` -- `src/models/stacking.py` -- Done: `StackingEnsemble.fit(X, y)` completes; `predict_proba` returns valid probabilities
- [ ] **T14**: Implement `src/models/meta_learner.py` -- `src/models/meta_learner.py` -- Done: meta-learner CV AUC > best individual base learner AUC

### P1 — Evaluation

- [ ] **T15**: Implement `src/evaluation/metrics.py` -- `src/evaluation/metrics.py` -- Done: `compute_metrics(y_true, y_pred, y_prob)` returns dict with all 6 metrics
- [ ] **T16**: Implement `src/evaluation/explainability.py` -- `src/evaluation/explainability.py` -- Done: `shap_summary(model, X)` saves PNG to `models_saved/`

### P1 — API

- [ ] **T17**: Implement `src/api/schemas.py` -- `src/api/schemas.py` -- Done: all 3 disease Pydantic input/output models defined and importable
- [ ] **T18**: Implement `src/api/predict.py` and `src/api/main.py` -- `src/api/` -- Done: `uvicorn src.api.main:app` starts; `POST /predict/heart` returns valid JSON
- [ ] **T19**: Write `tests/test_api.py` -- `tests/test_api.py` -- Done: `pytest tests/test_api.py` passes all tests

### P2 — Streamlit Dashboard

- [ ] **T20**: Implement `app/components/charts.py` and `app/components/sidebar.py` -- `app/components/` -- Done: chart functions callable with sample data, return Plotly Figure
- [ ] **T21**: Implement all 4 dashboard pages -- `app/pages/` -- Done: each page loads without error in `streamlit run`
- [ ] **T22**: Implement `app/streamlit_app.py` main entry -- `app/streamlit_app.py` -- Done: all pages reachable via sidebar navigation

### P2 — Containerization

- [ ] **T23**: Write `docker/Dockerfile.api` and `docker/Dockerfile.streamlit` -- `docker/` -- Done: `docker build` succeeds for both; containers start and respond to requests
- [ ] **T24**: Write `docker/docker-compose.yml` -- `docker/docker-compose.yml` -- Done: `docker-compose up` starts both services; Streamlit can reach FastAPI

### P2 — CI/CD and Cloud Deploy

- [ ] **T25**: Write `.github/workflows/ci.yml` -- `.github/workflows/ci.yml` -- Done: GitHub Actions run green on push
- [ ] **T26**: Configure Render web services and deploy -- Render dashboard -- Done: both URLs respond on the internet

### P3 — Polish and Tests

- [ ] **T27**: Write `tests/test_preprocessor.py`, `tests/test_stacking.py`, `tests/test_metrics.py` -- `tests/` -- Done: `pytest tests/` all pass
- [ ] **T28**: Write `README.md` with setup and deployment instructions -- `README.md` -- Done: a new contributor can run the project end-to-end following the README

---

## Dependency Graph

```
T01 → T02 → T03
T03 → T04 → T05 → T06 → T07
T07 → T11 → T13 → T14 → T15
T07 → T12 → T13
T14 → T16
T14 → T17 → T18 → T19
T05 → T08, T09, T10   (EDA notebooks can run after preprocessing is defined)
T18 → T21 → T22
T20 → T21
T22 → T23 → T24 → T25 → T26
T19 → T27
```

### Critical Path

T01 → T02 → T03 → T04 → T05 → T07 → T11 → T12 → T13 → T14 → T17 → T18 → T21 → T22 → T23 → T24 → T26

---

## Estimated Scope

| Metric | Value |
|--------|-------|
| Total tasks | 28 |
| P0 (critical) tasks | 7 |
| P1 (core) tasks | 12 |
| P2 (quality) tasks | 7 |
| P3 (polish) tasks | 2 |
| Estimated commits | ~35-45 |
| Estimated development time | 28-30 days (1 developer) |
| Python files to create | ~30 source files + 8 notebooks |

---

## Quality Checklist

- [ ] All functions in `src/` have type annotations and docstrings
- [ ] No raw `pd.DataFrame` operations duplicated across modules — all preprocessing through `MediSensePreprocessor`
- [ ] SMOTE applied only to training folds — never leaks into validation or test sets
- [ ] Random seed `42` set in `config.py` and propagated everywhere (`random_state=config.SEED`)
- [ ] Model artifacts versioned with dataset name (e.g., `models_saved/heart/stacking_ensemble.pkl`)
- [ ] All API endpoints have input validation ranges based on dataset statistics
- [ ] Docker images tested locally before cloud deploy
- [ ] Test coverage covers the preprocessor, stacking logic, and API routes
- [ ] `requirements.txt` uses pinned versions (`==`) not ranges (`>=`)
- [ ] `.env.example` documents all required environment variables with descriptions
```
