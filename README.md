# Hospital Intelligence Studio

An educational hospital analytics project with a FastAPI web application, dashboard, aggregate analytics, and a reproducible multi-parameter length-of-stay (LOS) prediction workflow. `main.py` is the Vercel entrypoint; the earlier Streamlit prototype remains in `app.py`.

## Stack

- FastAPI and Python for the app and JSON API
- pandas and scikit-learn for data preparation, evaluation, and prediction
- joblib for the fitted model pipeline
- HTML, CSS, and JavaScript for the dashboard
- Vercel Python Functions for hosting

## Run locally

Install dependencies and train the LOS model from the local training CSV:

```bash
python -m pip install -r requirements.txt
python train_los_model.py
uvicorn main:app --reload
```

Open `http://127.0.0.1:8000`. API documentation is at `/docs` and `/api/health` checks service status. The training command writes `models/los_admission_pipeline.joblib` and `models/los_model_metrics.json`.

The earlier Streamlit prototype is separate:

```bash
python -m pip install -r requirements-streamlit.txt
streamlit run app.py
```

## Model

The LOS endpoint predicts from age, gender, primary diagnosis, comorbidity count, planned surgery flag, insurance type, and social support. It uses numeric imputation/scaling, one-hot categorical encoding, and linear regression. Patient ID is excluded. ICU status, hospital-acquired infection, and discharge delay are excluded because they can be known only after admission and risk leakage for an admission-time forecast.

Evaluation on the supplied 10,000-row sample uses an 80/20 random holdout and five-fold cross-validation. Current holdout MAE is 2.18 days, RMSE 2.81 days, and R² 0.27; five-fold CV MAE is 2.16 ± 0.04 days. This is modest performance on an undocumented dataset, not evidence of performance at another hospital. The model does not forecast bed occupancy; that would require time-stamped admissions/discharges and available bed capacity.

The `scikit.py` compatibility entrypoint retrains and evaluates the same pipeline. `hospitalbedoccupancypredictor.py` runs an interactive local demo using the fitted artifact. Both are import-safe.

## Vercel

Connect the repository to Vercel and deploy the project root. `vercel.json` includes dashboard assets, the fitted model, evaluation metadata, and the legacy aggregate-demo CSV. The patient-level 10,000-row CSV is excluded from Vercel uploads. Do not commit patient-level source records to a public repository; deploy only the model artifact and aggregate evaluation metadata.

## Routes

- `/` — dashboard, insights, and prediction lab
- `/api/summary` — aggregate metrics from the legacy demonstration dataset
- `/api/predict` — auxiliary estimates from the legacy sample
- `/api/predict/length-of-stay` — admission-time LOS model
- `/api/model-info` — holdout and cross-validation metrics
- `/api/health` — service health
- `/docs` — interactive API documentation

## Limitations

Both datasets have undocumented provenance, collection method, and data dictionary. The legacy sample has 984 rows; the LOS sample has 10,000. Cross-validation and random holdout results do not establish performance on new patients or hospitals. This project is for learning only—not diagnosis, treatment, triage, staffing, bed allocation, reimbursement, or other real decisions. A credible operational model needs representative time-stamped data, feature-timing review, temporal and external validation, subgroup analysis, privacy controls, and domain review.
