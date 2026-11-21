"""Train and evaluate the admission-time length-of-stay model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "hospital_bed_occupancy_los_10000_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "los_admission_pipeline.joblib"
METRICS_PATH = MODEL_DIR / "los_model_metrics.json"
NUMERIC = ["Age", "Comorbidities_Count", "Surgery_Flag", "Social_Support"]
CATEGORICAL = ["Gender", "Primary_Diagnosis", "Insurance_Type"]
TARGET = "Length_of_Stay_Days"


def make_pipeline() -> Pipeline:
    prep = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    return Pipeline([("prepare", prep), ("model", LinearRegression())])


def main() -> None:
    df = pd.read_csv(DATA)
    required = set(NUMERIC + CATEGORICAL + [TARGET])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    evaluation_model = make_pipeline().fit(X_train, y_train)
    prediction = evaluation_model.predict(X_test)
    cv = cross_val_score(
        make_pipeline(), X, y, cv=KFold(n_splits=5, shuffle=True, random_state=42),
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    metrics = {
        "model": "Linear regression with scaled numeric and one-hot categorical features",
        "target": TARGET,
        "features": NUMERIC + CATEGORICAL,
        "excluded_features": {
            "Patient_ID": "Identifier; not a meaningful predictor",
            "ICU_Flag": "May be determined after admission",
            "Hospital_Acquired_Infection": "Occurs after admission",
            "Discharge_Delay_Days": "Known at or near discharge; leakage for admission-time forecast",
        },
        "rows": int(len(df)),
        "split": "80/20 random holdout, random_state=42",
        "holdout": {
            "mae_days": round(float(mean_absolute_error(y_test, prediction)), 4),
            "rmse_days": round(float(mean_squared_error(y_test, prediction) ** 0.5), 4),
            "r2": round(float(r2_score(y_test, prediction)), 4),
        },
        "five_fold_cv_mae_days_mean": round(float(-cv.mean()), 4),
        "five_fold_cv_mae_days_std": round(float(cv.std()), 4),
        "caveat": "Dataset provenance and feature timing are undocumented. Random-split results are educational and do not establish real-hospital performance.",
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(make_pipeline().fit(X, y), MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
