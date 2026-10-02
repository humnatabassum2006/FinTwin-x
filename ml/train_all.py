"""
Train every model in the platform and write a single metrics artefact.

    python -m ml.train_all

Outputs
    ml/models/artifacts/forecast_store.joblib   expense + income forecasting
    ml/models/artifacts/risk_model.joblib       calibrated stress-risk model
    ml/models/artifacts/risk_background.parquet SHAP background sample
    ml/models/artifacts/anomaly_model.joblib    transaction anomaly detector
    data/processed/model_metrics.json           evaluation report for the README
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from ml.anomaly_detection.anomaly import save_anomaly_model, train_anomaly_model
from ml.forecasting.cashflow_forecast import ForecastStore
from ml.models.registry import registry
from ml.risk.risk_model import save_risk_model, train_risk_model

log = get_logger("train")


def train_all(panel: pd.DataFrame | None = None, transactions: pd.DataFrame | None = None,
              dataset: pd.DataFrame | None = None, risk_algorithms=("xgboost", "lightgbm")) -> dict:
    t0 = time.time()
    panel = panel if panel is not None else pd.read_parquet(settings.PROCESSED_DIR / "monthly_panel.parquet")
    transactions = transactions if transactions is not None else pd.read_parquet(
        settings.PROCESSED_DIR / "transactions.parquet")
    dataset = dataset if dataset is not None else pd.read_parquet(
        settings.PROCESSED_DIR / "training_dataset.parquet")

    report: dict = {"trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "models": {}}

    # ------------------------------------------------------------------ forecasting
    log.info("=== forecasting models ===")
    store = ForecastStore().fit(panel, algorithm="xgboost")
    store.save()
    report["models"]["forecasting"] = store.metrics

    # ------------------------------------------------------------------ risk
    log.info("=== risk models ===")
    best, best_auc = None, -1.0
    for algo in risk_algorithms:
        try:
            rm = train_risk_model(dataset, algorithm=algo)
            report["models"].setdefault("risk", {})[algo] = {k: v for k, v in rm.metrics.items()
                                                             if k != "calibration_bins"}
            if rm.metrics["roc_auc"] > best_auc:
                best, best_auc = rm, rm.metrics["roc_auc"]
        except Exception as exc:                      # pragma: no cover
            log.warning("risk model %s failed: %s", algo, exc)
    if best is None:
        raise RuntimeError("no risk model could be trained")
    save_risk_model(best)
    bg = dataset[best.features].astype(float).sample(min(400, len(dataset)),
                                                     random_state=settings.RANDOM_SEED)
    bg.to_parquet(settings.MODEL_DIR / "risk_background.parquet", index=False)
    report["models"]["risk"]["selected"] = best.algorithm
    report["models"]["risk"]["calibration"] = best.metrics.get("calibration_bins", [])
    report["models"]["risk"]["top_features"] = best.metrics.get("top_features", {})

    # ------------------------------------------------------------------ anomaly
    log.info("=== anomaly detection ===")
    am = train_anomaly_model(transactions)
    save_anomaly_model(am)
    report["models"]["anomaly"] = am.metrics

    report["runtime_seconds"] = round(time.time() - t0, 2)
    report["registry"] = {k: [m["version"] for m in v] for k, v in registry.list_models().items()}
    (settings.PROCESSED_DIR / "model_metrics.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    log.info("training complete in %.1fs → %s", time.time() - t0,
             settings.PROCESSED_DIR / "model_metrics.json")
    return report


def main() -> None:
    report = train_all()
    print(json.dumps({
        "forecast_mae_h1": report["models"]["forecasting"]["expense"]["1"]["mae"],
        "forecast_mape_h1": report["models"]["forecasting"]["expense"]["1"]["mape_pct"],
        "risk_auc": report["models"]["risk"][report["models"]["risk"]["selected"]]["roc_auc"],
        "anomaly_precision@1000": report["models"]["anomaly"].get("precision@1000"),
        "runtime": report["runtime_seconds"],
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
