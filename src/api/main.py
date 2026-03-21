from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.health import router as health_router
from src.api.schemas import (
    HeartDiseaseInput,
    DiabetesInput,
    LiverDiseaseInput,
    RiskPredictionOutput,
)
from src.api.predict import predict_risk
from src.config import DATASETS

app = FastAPI(
    title="MediSense AI",
    description="Intelligent Health Risk Prediction System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router)

INPUT_SCHEMAS = {
    "heart": HeartDiseaseInput,
    "diabetes": DiabetesInput,
    "liver": LiverDiseaseInput,
}


@app.get("/models")
def list_models():
    return {"models": list(DATASETS.keys())}


@app.post("/predict/{disease}", response_model=RiskPredictionOutput)
def predict(disease: str, data: dict):
    if disease not in DATASETS:
        raise HTTPException(
            status_code=404, detail=f"Disease model '{disease}' not found"
        )

    schema_cls = INPUT_SCHEMAS[disease]
    try:
        validated = schema_cls(**data)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))

    result = predict_risk(disease, validated.model_dump())
    return result


@app.get("/predict/{disease}/features")
def get_features(disease: str):
    if disease not in INPUT_SCHEMAS:
        raise HTTPException(
            status_code=404, detail=f"Disease model '{disease}' not found"
        )

    schema = INPUT_SCHEMAS[disease].model_json_schema()
    return {"disease": disease, "features": schema.get("properties", {})}
