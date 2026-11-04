"""Small-data modeling workflow for the hospital analytics prototype.

Predictions are for learning and exploration only. The bundled dataset is
synthetic-looking and is not suitable for clinical or operational decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURES = ["Age", "Gender", "Condition", "Procedure"]
CLASSIFICATION_TARGETS = ["Readmission", "Outcome", "Satisfaction"]
REGRESSION_TARGETS = ["Cost", "Length_of_Stay"]
TARGETS = CLASSIFICATION_TARGETS + REGRESSION_TARGETS


@dataclass
class TargetResult:
    target: str
    task: str
    model_name: str
    model: Pipeline
    metrics: dict[str, float]
    comparison: list[dict[str, float | str]]
    classes: list[Any] | None = None


def _preprocessor() -> ColumnTransformer:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numeric", numeric, ["Age"]),
        ("categorical", categorical, ["Gender", "Condition", "Procedure"]),
    ])


def _classification_candidates() -> dict[str, Any]:
    return {
        "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced", random_state=42, n_jobs=-1),
    }


def _regression_candidates() -> dict[str, Any]:
    return {
        "Ridge regression": Ridge(alpha=1.0),
        "Random forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
        "Extra trees": ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    }


def _folds(y: pd.Series, task: str):
    n_splits = min(5, int(y.value_counts().min())) if task == "classification" else 5
    if n_splits < 2:
        raise ValueError("This target needs at least two examples in every class.")
    return (StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42) if task == "classification"
            else KFold(n_splits=n_splits, shuffle=True, random_state=42))


def _score_classification(y: pd.Series, pred: np.ndarray, prob: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    try:
        if len(labels) == 2:
            result["roc_auc"] = float(roc_auc_score(y, prob[:, 1]))
        else:
            result["roc_auc"] = float(roc_auc_score(y, prob, multi_class="ovr", labels=labels, average="macro"))
    except ValueError:
        result["roc_auc"] = float("nan")
    return result


def train_target_models(data: pd.DataFrame) -> dict[str, TargetResult]:
    """Compare three model families with shuffled 5-fold out-of-fold estimates."""
    missing = set(FEATURES + TARGETS) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")
    frame = data[FEATURES + TARGETS].dropna().copy()
    if len(frame) < 30:
        raise ValueError("At least 30 complete records are required for this prototype workflow.")

    x = frame[FEATURES]
    results: dict[str, TargetResult] = {}
    for target in TARGETS:
        y = frame[target]
        task = "classification" if target in CLASSIFICATION_TARGETS else "regression"
        candidates = _classification_candidates() if task == "classification" else _regression_candidates()
        folds = _folds(y, task)
        comparison: list[dict[str, float | str]] = []
        fitted: dict[str, tuple[Pipeline, dict[str, float], float]] = {}
        labels = np.sort(y.unique()) if task == "classification" else None

        for name, estimator in candidates.items():
            pipe = Pipeline([("prepare", _preprocessor()), ("model", estimator)])
            if task == "classification":
                pred = cross_val_predict(pipe, x, y, cv=folds, method="predict", n_jobs=1)
                prob = cross_val_predict(pipe, x, y, cv=folds, method="predict_proba", n_jobs=1)
                scores = _score_classification(y, pred, prob, labels)
                selection_score = scores["macro_f1"]
            else:
                pred = cross_val_predict(pipe, x, y, cv=folds, method="predict", n_jobs=1)
                scores = {
                    "mae": float(mean_absolute_error(y, pred)),
                    "rmse": float(np.sqrt(mean_squared_error(y, pred))),
                    "r2": float(r2_score(y, pred)),
                }
                selection_score = -scores["mae"]
            comparison.append({"model": name, **scores})
            fitted[name] = (pipe, scores, selection_score)

        best_name = max(fitted, key=lambda name: fitted[name][2])
        best_pipe, best_scores, _ = fitted[best_name]
        best_pipe.fit(x, y)
        results[target] = TargetResult(
            target=target,
            task=task,
            model_name=best_name,
            model=best_pipe,
            metrics=best_scores,
            comparison=comparison,
            classes=list(labels) if labels is not None else None,
        )
    return results


def predict_all(results: dict[str, TargetResult], patient: dict[str, Any]) -> dict[str, Any]:
    row = pd.DataFrame([{feature: patient[feature] for feature in FEATURES}])
    output: dict[str, Any] = {}
    for target, result in results.items():
        prediction = result.model.predict(row)[0]
        item: dict[str, Any] = {"prediction": prediction}
        if result.task == "classification":
            probabilities = result.model.predict_proba(row)[0]
            item["probabilities"] = dict(zip(result.model.named_steps["model"].classes_, probabilities))
        output[target] = item
    return output


def feature_influence(result: TargetResult) -> pd.DataFrame:
    """Aggregate encoded model weights/importances back to original input fields."""
    estimator = result.model.named_steps["model"]
    names = result.model.named_steps["prepare"].get_feature_names_out()
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        values = np.mean(np.abs(np.asarray(estimator.coef_, dtype=float)), axis=0)
    else:
        return pd.DataFrame(columns=["Input", "Relative influence"])

    aggregated = {feature: 0.0 for feature in FEATURES}
    for name, value in zip(names, values):
        # ColumnTransformer names are prefixed with numeric__ or categorical__.
        raw = name.split("__", 1)[-1]
        parent = "Age" if raw == "Age" else next((field for field in FEATURES if raw.startswith(field + "_")), None)
        if parent:
            aggregated[parent] += float(value)
    total = sum(aggregated.values())
    if total:
        aggregated = {key: value / total for key, value in aggregated.items()}
    return (pd.DataFrame([{"Input": key, "Relative influence": value} for key, value in aggregated.items()])
            .sort_values("Relative influence", ascending=False, ignore_index=True))
