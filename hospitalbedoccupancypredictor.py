"""Interactive local demo using the same fitted pipeline as the web app."""
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "models" / "los_admission_pipeline.joblib"


def main() -> None:
    if not MODEL.exists():
        raise SystemExit("Model not found. Run: python train_los_model.py")
    row = {
        "Age": float(input("Age: ")),
        "Gender": input("Gender (Female/Male): ").strip(),
        "Primary_Diagnosis": input("Primary diagnosis: ").strip(),
        "Comorbidities_Count": float(input("Comorbidities count: ")),
        "Surgery_Flag": int(input("Surgery planned? (0/1): ")),
        "Insurance_Type": input("Insurance (Government/Private/Uninsured): ").strip(),
        "Social_Support": int(input("Social support available? (0/1): ")),
    }
    estimate = float(joblib.load(MODEL).predict(pd.DataFrame([row]))[0])
    print(f"Estimated length of stay: {max(0.0, estimate):.1f} days")
    print("Educational sample estimate only; not validated for hospital decisions.")


if __name__ == "__main__":
    main()
