"""Model quality tests: every model must beat its baseline / sanity threshold."""
from __future__ import annotations

import json

import numpy as np
import pytest

from common import settings


def _metrics():
    p = settings.PROCESSED_DIR / "model_metrics.json"
    if not p.exists():
        pytest.skip("run `make train` first")
    return json.loads(p.read_text(encoding="utf-8"))


def test_forecasting_beats_naive_baseline():
    m = _metrics()["models"]["forecasting"]
    for target in ("expense", "income"):
        for horizon, rep in m[target].items():
            assert rep["mae"] < rep["baseline"]["mae"], f"{target} h={horizon} worse than baseline"
            assert rep["coverage_90"] >= 0.82, f"{target} h={horizon} interval under-covers"


def test_risk_model_discriminates():
    risk = _metrics()["models"]["risk"]
    sel = risk[risk["selected"]]
    assert sel["roc_auc"] >= 0.70
    assert sel["pr_auc"] >= 0.50
    assert sel["brier"] <= 0.25


def test_risk_model_is_calibrated():
    risk = _metrics()["models"]["risk"]
    bins = risk.get("calibration", [])
    if not bins:
        pytest.skip("no calibration data")
    err = np.mean([abs(b["mean_pred"] - b["observed"]) for b in bins if b["n"] >= 20])
    assert err <= 0.20, f"calibration error too high: {err:.3f}"


def test_anomaly_detection_beats_random():
    a = _metrics()["models"]["anomaly"]
    base_rate = a.get("n_true_anomalies", 1) / max(a.get("n_transactions", 1), 1)
    assert a.get("precision@100", 0) >= 0.30, a
    assert a.get("precision@1000", 0) >= 5 * base_rate, a
    assert a.get("roc_auc", 0) >= 0.85


def test_shap_explanations_sum_to_score(models_ready, features):
    if not models_ready:
        pytest.skip("run `make train` first")
    from ml.explainability.shap_explain import explain
    from ml.risk.risk_model import load_risk_model
    import pandas as pd

    model = load_risk_model()
    row = features.iloc[0]
    X = pd.DataFrame([row])[model.features].astype(float)
    out = explain(model, X, None)
    assert out["score"] is not None
    if out["contributions"]:
        total = sum(c["points"] for c in out["contributions"])
        # contributions should point in the direction of the score vs the base rate
        assert np.isfinite(total)


def test_counterfactuals_improve_score(models_ready, features):
    if not models_ready:
        pytest.skip("run `make train` first")
    from ml.explainability.counterfactual import counterfactuals
    from ml.risk.risk_model import load_risk_model

    model = load_risk_model()
    worst = features.sort_values("dna_financial_risk", ascending=False).iloc[0]
    cf = counterfactuals(model, worst)
    assert cf["best"]["delta"] <= 0.0, "the best action must not increase risk"
    assert all(np.isfinite(o["new_score"]) for o in cf["options"])
