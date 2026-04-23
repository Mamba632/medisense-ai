DISEASE_INFO = {
    "heart": {
        "title": "Heart Disease",
        "icon": "❤️",
        "color": "#E63946",
        "description": (
            "Predict risk of heart disease based on clinical indicators like "
            "blood pressure, cholesterol, and heart rate."
        ),
    },
    "diabetes": {
        "title": "Diabetes",
        "icon": "🩸",
        "color": "#F4A261",
        "description": (
            "Assess diabetes risk using indicators such as glucose level, BMI, "
            "insulin, and family history."
        ),
    },
    "liver": {
        "title": "Liver Disease",
        "icon": "🫁",
        "color": "#2A9D8F",
        "description": (
            "Evaluate liver disease risk based on enzyme levels, bilirubin, "
            "and protein indicators."
        ),
    },
}

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

FEATURE_INPUT_META = {
    "age": {
        "min": 1,
        "max": 120,
        "step": 1,
        "placeholder": "52",
        "hint": "1-120 years",
    },
    "trestbps": {
        "min": 60,
        "max": 250,
        "step": 1,
        "placeholder": "120",
        "hint": "Typical resting systolic range: 90-140 mm Hg",
    },
    "chol": {
        "min": 100,
        "max": 600,
        "step": 1,
        "placeholder": "200",
        "hint": "Total cholesterol in mg/dl",
    },
    "thalach": {
        "min": 50,
        "max": 220,
        "step": 1,
        "placeholder": "150",
        "hint": "Maximum heart rate",
    },
    "oldpeak": {
        "min": 0,
        "max": 7,
        "step": 0.1,
        "placeholder": "1.2",
        "hint": "ST depression value",
    },
    "ca": {
        "min": 0,
        "max": 4,
        "step": 1,
        "placeholder": "0",
        "hint": "0-4 vessels",
    },
    "thal": {
        "min": 0,
        "max": 3,
        "step": 1,
        "placeholder": "2",
        "hint": "0-3 category",
    },
    "Pregnancies": {
        "min": 0,
        "max": 20,
        "step": 1,
        "placeholder": "2",
        "hint": "Use 0 if not applicable",
    },
    "Glucose": {
        "min": 0,
        "max": 250,
        "step": 1,
        "placeholder": "110",
        "hint": "Plasma glucose concentration",
    },
    "BloodPressure": {
        "min": 0,
        "max": 150,
        "step": 1,
        "placeholder": "72",
        "hint": "Diastolic blood pressure",
    },
    "SkinThickness": {
        "min": 0,
        "max": 100,
        "step": 1,
        "placeholder": "25",
        "hint": "Triceps skin fold thickness",
    },
    "Insulin": {
        "min": 0,
        "max": 900,
        "step": 1,
        "placeholder": "80",
        "hint": "2-hour serum insulin",
    },
    "BMI": {
        "min": 0,
        "max": 70,
        "step": 0.1,
        "placeholder": "24.5",
        "hint": "Body mass index",
    },
    "DiabetesPedigreeFunction": {
        "min": 0,
        "max": 3,
        "step": 0.001,
        "placeholder": "0.45",
        "hint": "Family-history score",
    },
    "Age": {
        "min": 1,
        "max": 120,
        "step": 1,
        "placeholder": "45",
        "hint": "1-120 years",
    },
    "Total_Bilirubin": {
        "min": 0,
        "max": 80,
        "step": 0.1,
        "placeholder": "1.0",
        "hint": "mg/dl",
    },
    "Direct_Bilirubin": {
        "min": 0,
        "max": 20,
        "step": 0.1,
        "placeholder": "0.3",
        "hint": "mg/dl",
    },
    "Alkaline_Phosphotase": {
        "min": 50,
        "max": 2500,
        "step": 1,
        "placeholder": "190",
        "hint": "IU/L",
    },
    "Alamine_Aminotransferase": {
        "min": 5,
        "max": 2500,
        "step": 1,
        "placeholder": "35",
        "hint": "ALT / SGPT in IU/L",
    },
    "Aspartate_Aminotransferase": {
        "min": 5,
        "max": 5000,
        "step": 1,
        "placeholder": "40",
        "hint": "AST / SGOT in IU/L",
    },
    "Total_Protiens": {
        "min": 2,
        "max": 10,
        "step": 0.1,
        "placeholder": "6.8",
        "hint": "g/dl",
    },
    "Albumin": {
        "min": 0.5,
        "max": 6,
        "step": 0.1,
        "placeholder": "3.8",
        "hint": "g/dl",
    },
    "Albumin_and_Globulin_Ratio": {
        "min": 0,
        "max": 3,
        "step": 0.1,
        "placeholder": "1.2",
        "hint": "A/G ratio",
    },
}

DROPDOWN_OPTIONS = {
    "sex": [("0", "Female"), ("1", "Male")],
    "cp": [
        ("1", "Typical Angina"),
        ("2", "Atypical Angina"),
        ("3", "Non-anginal Pain"),
        ("4", "Asymptomatic"),
    ],
    "fbs": [("0", "No"), ("1", "Yes")],
    "restecg": [("0", "Normal"), ("1", "ST-T Abnormality"), ("2", "LV Hypertrophy")],
    "exang": [("0", "No"), ("1", "Yes")],
    "slope": [("1", "Upsloping"), ("2", "Flat"), ("3", "Downsloping")],
    "ca": [
        ("0", "0 vessels"),
        ("1", "1 vessel"),
        ("2", "2 vessels"),
        ("3", "3 vessels"),
    ],
    "thal": [("3", "Normal"), ("6", "Fixed Defect"), ("7", "Reversible Defect")],
    "Gender": [("Male", "Male"), ("Female", "Female")],
}

ASSESSMENT_TIPS = {
    "heart": [
        "Use recent blood pressure and cholesterol readings when available.",
        "If ECG or thalassemia values are unclear, check your lab report first.",
        "Chest pain, breathlessness, or fainting needs medical attention regardless of the score.",
    ],
    "diabetes": [
        "Recent fasting or post-meal glucose values give a more meaningful estimate.",
        "BMI, insulin, and family-history score can strongly affect this model.",
        "Very low zero values may mean missing clinical data, not a healthy value.",
    ],
    "liver": [
        "Use values from the same liver-function test report when possible.",
        "Enter bilirubin, ALT, AST, albumin, and A/G ratio with their report units.",
        "Yellowing eyes, severe abdominal pain, or confusion needs urgent medical care.",
    ],
}

NEXT_STEPS = {
    "Low": [
        "Save this result as a baseline for future comparisons.",
        "Keep routine check-ups and healthy habits consistent.",
        "Repeat the assessment when new lab values are available.",
    ],
    "Medium": [
        "Discuss this result with a healthcare professional during a planned visit.",
        "Track the indicators that were outside the usual range.",
        "Repeat the assessment after updated tests or lifestyle changes.",
    ],
    "High": [
        "Arrange medical review as soon as possible and share this report.",
        "Do not delay care if symptoms are strong, sudden, or worsening.",
        "Keep a copy of recent lab reports, medicines, and symptoms ready.",
    ],
}

REPORT_CHECKLIST = [
    "Risk score and disease type",
    "Submitted health indicators",
    "Recent lab report or prescription",
    "Symptoms, duration, and current medicines",
]
