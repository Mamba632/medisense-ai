from pydantic import BaseModel, Field


class HeartDiseaseInput(BaseModel):
    age: float = Field(..., ge=1, le=120)
    sex: float = Field(..., ge=0, le=1)
    cp: float = Field(..., ge=0, le=3)
    trestbps: float = Field(..., ge=60, le=250)
    chol: float = Field(..., ge=100, le=600)
    fbs: float = Field(..., ge=0, le=1)
    restecg: float = Field(..., ge=0, le=2)
    thalach: float = Field(..., ge=50, le=220)
    exang: float = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=7)
    slope: float = Field(..., ge=0, le=2)
    ca: float = Field(..., ge=0, le=4)
    thal: float = Field(..., ge=0, le=3)


class DiabetesInput(BaseModel):
    Pregnancies: float = Field(..., ge=0, le=20)
    Glucose: float = Field(..., ge=0, le=250)
    BloodPressure: float = Field(..., ge=0, le=150)
    SkinThickness: float = Field(..., ge=0, le=100)
    Insulin: float = Field(..., ge=0, le=900)
    BMI: float = Field(..., ge=0, le=70)
    DiabetesPedigreeFunction: float = Field(..., ge=0, le=3)
    Age: float = Field(..., ge=1, le=120)


class LiverDiseaseInput(BaseModel):
    Age: float = Field(..., ge=1, le=120)
    Gender: str = Field(..., pattern="^(Male|Female)$")
    Total_Bilirubin: float = Field(..., ge=0, le=80)
    Direct_Bilirubin: float = Field(..., ge=0, le=20)
    Alkaline_Phosphotase: float = Field(..., ge=50, le=2500)
    Alamine_Aminotransferase: float = Field(..., ge=5, le=2500)
    Aspartate_Aminotransferase: float = Field(..., ge=5, le=5000)
    Total_Protiens: float = Field(..., ge=2, le=10)
    Albumin: float = Field(..., ge=0.5, le=6)
    Albumin_and_Globulin_Ratio: float = Field(..., ge=0, le=3)


class RiskPredictionOutput(BaseModel):
    disease: str
    risk_probability: float
    risk_label: str
    top_risk_factors: list[dict] = []
    model_version: str = "1.0.0"
