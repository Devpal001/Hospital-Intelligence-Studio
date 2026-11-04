# Hospital Intelligence Studio

An educational hospital analytics project with a FastAPI web application, a responsive dashboard, aggregate analytics, and two prediction workflows. The Vercel production entrypoint is `main.py`; the earlier Streamlit prototype remains available locally in `app.py`.

## Stack

- **FastAPI + Python 3.12** for the web app and JSON API
- **pandas + scikit-learn** for data summaries and prediction
- **HTML, CSS, and JavaScript** for the responsive dashboard UI
- **Vercel Python Functions** for hosting

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. The API schema is at `/docs`; `/api/health` checks service availability.

To run the original Streamlit prototype instead:

```bash
python -m pip install -r requirements-streamlit.txt
streamlit run app.py
```

## Vercel deployment

The project is configured for Vercel's Python runtime. Connect the repository to Vercel and deploy the project root. Vercel reads the Python version from `.python-version`, installs `requirements.txt`, and uses `main.py` as the FastAPI entrypoint. `vercel.json` includes the HTML, CSS, JavaScript, and CSV assets needed by the function.

For CLI deployment, install and authenticate the Vercel CLI, then run:

```bash
vercel
vercel --prod
```

## Web routes

- `/` — dashboard, insights, and prediction lab
- `/api/summary` — aggregate metrics only
- `/api/predict` — five scenario-based estimates from the 984-row demonstration dataset
- `/api/predict/length-of-stay` — the seven-factor linear model from `hospitalbedoccupancypredictor.py`
- `/api/health` — service health check
- `/docs` — interactive API documentation

The hosted API deliberately does not return patient-level records. Do not upload real patient data to this public demo.

## Prediction workflow

The demonstration prediction endpoint accepts age, gender, condition, and procedure and returns estimates for readmission, outcome, cost, length of stay, and satisfaction. It uses one Extra Trees model per target, trained once per warm serverless function instance and cached in memory. Unknown category combinations are rejected. Class probabilities are displayed as model estimates, not calibrated real-world risk.

The separate seven-factor endpoint uses the saved linear regression coefficients in `hospitalbedoccupancypredictor.py` for length-of-stay estimation. That script references `hospital_bed_occupancy_los_10000_clean.csv`, which is not included in this workspace. The coefficient model therefore cannot be independently retrained or evaluated here. The UI asks for the source encoding for social support rather than assuming a scale.

## Data limitations

The bundled CSV contains 984 rows but has no documented source, collection method, data dictionary, currency, or length-of-stay unit. Several values are unusually regular within conditions, suggesting the data may be synthetic or constructed. Cross-validation results on this dataset do not establish performance on new hospitals or patients.

This project is for learning and demonstration only. Do not use it for diagnosis, treatment, triage, staffing, bed allocation, reimbursement, or other real decisions. A credible hospital operations model would need representative, time-stamped data, documented labels and units, leakage review, temporal and external validation, subgroup analysis, privacy controls, and domain review.
