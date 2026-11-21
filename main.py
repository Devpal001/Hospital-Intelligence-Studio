from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "hospital data analysis.csv"
LOS_MODEL_PATH = ROOT / "models" / "los_admission_pipeline.joblib"
LOS_METRICS_PATH = ROOT / "models" / "los_model_metrics.json"
FEATURES = ["Age", "Gender", "Condition", "Procedure"]
CLASSIFICATION_TARGETS = ["Readmission", "Outcome", "Satisfaction"]
REGRESSION_TARGETS = ["Cost", "Length_of_Stay"]

app = FastAPI(
    title="Hospital Intelligence Studio",
    description="Exploratory hospital analytics and educational prediction API.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


class PatientScenario(BaseModel):
    age: int = Field(ge=18, le=100)
    gender: str = Field(min_length=1, max_length=30)
    condition: str = Field(min_length=1, max_length=80)
    procedure: str = Field(min_length=1, max_length=120)


class LengthOfStayScenario(BaseModel):
    age: float = Field(ge=1, le=120)
    gender: str = Field(min_length=1, max_length=30)
    primary_diagnosis: str = Field(min_length=1, max_length=100)
    comorbidities_count: float = Field(ge=0, le=50)
    surgery_flag: int = Field(ge=0, le=1)
    insurance_type: str = Field(min_length=1, max_length=40)
    social_support: int = Field(ge=0, le=1)


@lru_cache(maxsize=1)
def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise RuntimeError("The configured hospital dataset is missing.")
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + CLASSIFICATION_TARGETS + REGRESSION_TARGETS + ["Patient_ID"])
    missing = required - set(data.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    return data


def get_preprocessor() -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, ["Age"]),
        ("categorical", categorical, ["Gender", "Condition", "Procedure"]),
    ])


@lru_cache(maxsize=1)
def get_models() -> dict[str, Pipeline]:
    """Train small cached demo models once per warm function instance."""
    data = load_data()
    features = data[FEATURES]
    models: dict[str, Pipeline] = {}
    for target in CLASSIFICATION_TARGETS:
        model = Pipeline([
            ("prepare", get_preprocessor()),
            ("model", ExtraTreesClassifier(
                n_estimators=120,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=42,
                n_jobs=1,
            )),
        ])
        model.fit(features, data[target])
        models[target] = model
    for target in REGRESSION_TARGETS:
        model = Pipeline([
            ("prepare", get_preprocessor()),
            ("model", ExtraTreesRegressor(
                n_estimators=120,
                min_samples_leaf=4,
                random_state=42,
                n_jobs=1,
            )),
        ])
        model.fit(features, data[target])
        models[target] = model
    return models


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home() -> str:
    page = ROOT / "templates" / "index.html"
    return page.read_text(encoding="utf-8")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "hospital-intelligence"}


@app.get("/api/summary")
def summary() -> dict[str, Any]:
    data = load_data()
    condition_summary = data.groupby("Condition", as_index=False).agg(
        patients=("Patient_ID", "size"),
        average_cost=("Cost", "mean"),
        average_stay=("Length_of_Stay", "mean"),
        readmission_rate=("Readmission", lambda values: float((values == "Yes").mean())),
        average_satisfaction=("Satisfaction", "mean"),
        recovery_rate=("Outcome", lambda values: float((values == "Recovered").mean())),
    )
    return {
        "record_count": int(len(data)),
        "average_cost": float(data["Cost"].mean()),
        "average_stay": float(data["Length_of_Stay"].mean()),
        "readmission_rate": float((data["Readmission"] == "Yes").mean()),
        "average_satisfaction": float(data["Satisfaction"].mean()),
        "recovery_rate": float((data["Outcome"] == "Recovered").mean()),
        "age_average": float(data["Age"].mean()),
        "gender_counts": {str(k): int(v) for k, v in data["Gender"].value_counts().items()},
        "conditions": sorted(str(value) for value in data["Condition"].dropna().unique()),
        "procedures_by_condition": {
            str(condition): sorted(str(value) for value in group["Procedure"].dropna().unique())
            for condition, group in data.groupby("Condition")
        },
        "condition_summary": condition_summary.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.post("/api/predict")
def predict(scenario: PatientScenario) -> dict[str, Any]:
    data = load_data()
    # Only accept feature combinations represented in the bundled dataset.
    valid_gender = set(data["Gender"].dropna().astype(str))
    if scenario.gender not in valid_gender:
        raise HTTPException(status_code=422, detail="Choose a gender available in this dataset.")
    valid_pairs = set(zip(data["Condition"].astype(str), data["Procedure"].astype(str)))
    if (scenario.condition, scenario.procedure) not in valid_pairs:
        raise HTTPException(status_code=422, detail="Choose a procedure associated with the selected condition.")
    row = pd.DataFrame([{
        "Age": scenario.age,
        "Gender": scenario.gender,
        "Condition": scenario.condition,
        "Procedure": scenario.procedure,
    }])
    models = get_models()
    result: dict[str, Any] = {}
    for target in CLASSIFICATION_TARGETS:
        model = models[target]
        classes = model.named_steps["model"].classes_
        probabilities = model.predict_proba(row)[0]
        result[target] = {
            "prediction": str(model.predict(row)[0]),
            "probabilities": {str(label): float(value) for label, value in zip(classes, probabilities)},
        }
    for target in REGRESSION_TARGETS:
        result[target] = {"prediction": float(models[target].predict(row)[0])}
    return {"predictions": result, "model": "Extra Trees", "inputs": scenario.model_dump()}


@app.post("/api/predict/length-of-stay")
def predict_length_of_stay(scenario: LengthOfStayScenario) -> dict[str, Any]:
    """Predict LOS from features available at admission using the trained pipeline."""
    if not LOS_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="LOS model artifact is unavailable. Train it with train_los_model.py.")
    model = load_los_model()
    row = pd.DataFrame([{
        "Age": scenario.age,
        "Gender": scenario.gender,
        "Primary_Diagnosis": scenario.primary_diagnosis,
        "Comorbidities_Count": scenario.comorbidities_count,
        "Surgery_Flag": scenario.surgery_flag,
        "Insurance_Type": scenario.insurance_type,
        "Social_Support": scenario.social_support,
    }])
    estimate = float(model.predict(row)[0])
    return {
        "prediction_days": max(0.0, estimate),
        "model": "Admission-time multivariate linear regression",
        "inputs": scenario.model_dump(),
        "note": "Educational estimate from an undocumented sample; not validated for clinical or operational decisions.",
    }


@lru_cache(maxsize=1)
def load_los_model() -> Any:
    return joblib.load(LOS_MODEL_PATH)


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    if not LOS_METRICS_PATH.exists():
        raise HTTPException(status_code=503, detail="Model evaluation metadata is unavailable.")
    return json.loads(LOS_METRICS_PATH.read_text(encoding="utf-8"))
