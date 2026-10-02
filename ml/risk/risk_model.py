"""
Model C — FinTwin Risk Score (0–100).

Supervised: P(financial stress within the next 6 months) learned from the
labelled training dataset (features strictly before the outcome window).

The predicted probability is calibrated (isotonic regression) and mapped to a
0–100 score, so "38" always means the same thing: a ~38% modelled chance of
hitting stress in the next two quarters. A deterministic rule-based score is
kept alongside it for explainability and for cold-start users with short
histories.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common import settings
from common.logging_utils import get_logger
from ml.evaluation.metrics import calibration_bins, classification_report
from ml.models.registry import registry

log = get_logger("risk")

EXCLUDE = {"user_id", "as_of", "persona", "city", "income_type", "employment_type",
           "financial_goal", "risk_preference", "segment", "segment_label", "segment_id",
           "stress_label", "min_cash", "neg_months", "outcome_months"}


@dataclass
class RiskModel:
    model: object
    calibrator: IsotonicRegression
    features: list
    metrics: dict = field(default_factory=dict)
    algorithm: str = "xgboost"

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X = X[self.features]
        raw = _raw_predict(self.model, X, self.algorithm)
        return np.clip(self.calibrator.predict(raw), 0.0, 1.0)

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X) * 100).round(1)


def _xgb_dmatrix(X):  # local import kept tiny for pickle-friendliness
    import xgboost as xgb
    return xgb.DMatrix(X)


def _raw_predict(model, X: pd.DataFrame, algorithm: str) -> np.ndarray:
    """Probability of the positive class for XGBoost or LightGBM boosters."""
    if algorithm == "xgboost":
        import xgboost as xgb
        return np.asarray(model.predict(xgb.DMatrix(X)), dtype=float)
    return np.asarray(model.predict(X), dtype=float)


def _fit_model(name: str, X: pd.DataFrame, y: np.ndarray):
    if name == "xgboost":
        import xgboost as xgb
        params = dict(objective="binary:logistic", eval_metric="auc", max_depth=5, eta=0.05,
                      subsample=0.85, colsample_bytree=0.8, min_child_weight=20,
                      reg_lambda=2.0, tree_method="hist")
        dtrain = xgb.DMatrix(X, label=y)
        return xgb.train(params, dtrain, num_boost_round=400), "xgboost"
    import lightgbm as lgb
    params = dict(objective="binary", learning_rate=0.05, num_leaves=31, feature_fraction=0.8,
                  bagging_fraction=0.85, bagging_freq=1, min_data_in_leaf=40, verbose=-1)
    return lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=400), "lightgbm"


def train_risk_model(dataset: pd.DataFrame, algorithm: str = "xgboost") -> RiskModel:
    features = [c for c in dataset.columns if c not in EXCLUDE and pd.api.types.is_numeric_dtype(dataset[c])]
    X, y = dataset[features].astype(float), dataset["stress_label"].astype(int).to_numpy()

    X_tr_full, X_te, y_tr_full, y_te = train_test_split(
        X, y, test_size=0.25, random_state=settings.RANDOM_SEED, stratify=y)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr_full, y_tr_full, test_size=0.3, random_state=settings.RANDOM_SEED, stratify=y_tr_full)

    model, algo = _fit_model(algorithm, X_tr, y_tr)
    raw_cal = _raw_predict(model, X_cal, algo)
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_cal, y_cal)

    prob = np.clip(calibrator.predict(_raw_predict(model, X_te, algo)), 0, 1)
    metrics = classification_report(y_te, prob)
    metrics["calibration_bins"] = calibration_bins(y_te, prob)

    importances = _feature_importance(model, features, algo)
    metrics["top_features"] = dict(list(importances.items())[:12])

    rm = RiskModel(model=model, calibrator=calibrator, features=features, metrics=metrics, algorithm=algo)
    registry.register(
        name=f"risk_model_{algo}", model=rm, algorithm=algo, task="binary_classification",
        metrics={k: v for k, v in metrics.items() if k != "calibration_bins"},
        params={"n_estimators": 400, "max_depth": 5, "calibration": "isotonic"},
        features=features, dataset="training_dataset", rows=int(len(X_tr)),
        notes="P(financial stress in the next 6 months), isotonic-calibrated.",
    )
    log.info("risk model (%s): AUC=%.3f PR-AUC=%.3f F1=%.3f Brier=%.3f",
             algo, metrics["roc_auc"], metrics["pr_auc"], metrics["f1"], metrics["brier"])
    return rm


def _feature_importance(model, features: list, algo: str) -> dict:
    if algo == "xgboost":
        vals = model.get_score(importance_type="gain")
        imp = {f: float(vals.get(f, 0.0)) for f in features}
    else:
        imp = dict(zip(features, model.feature_importance("gain").tolist()))
    return dict(sorted(imp.items(), key=lambda kv: -kv[1]))


def save_risk_model(rm: RiskModel, path: Path | None = None) -> Path:
    import joblib
    path = Path(path or settings.MODEL_DIR / "risk_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(rm, path)
    return path


def load_risk_model(path: Path | None = None):
    import joblib
    path = Path(path or settings.MODEL_DIR / "risk_model.joblib")
    return joblib.load(path) if path.exists() else None


def risk_band(score: float) -> str:
    if score < 25:
        return "LOW"
    if score < 50:
        return "MODERATE"
    if score < 75:
        return "ELEVATED"
    return "HIGH"
