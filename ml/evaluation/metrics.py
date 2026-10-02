"""Evaluation metrics for forecasting, classification, anomaly detection and simulation."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             r2_score, roc_auc_score)


def _safe_prf(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Precision / recall / F1 computed directly (zero-division safe everywhere)."""
    tp = float(np.sum((pred == 1) & (y == 1)))
    fp = float(np.sum((pred == 1) & (y == 0)))
    fn = float(np.sum((pred == 0) & (y == 1)))
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --------------------------------------------------------------------------------------
# regression / forecasting
# --------------------------------------------------------------------------------------
def mae(y, yhat) -> float:
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(yhat))))


def rmse(y, yhat) -> float:
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(yhat)) ** 2)))


def mape(y, yhat, eps: float = 1e-6) -> float:
    y, yhat = np.asarray(y, dtype=float), np.asarray(yhat, dtype=float)
    mask = np.abs(y) > eps
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y[mask] - yhat[mask]) / y[mask])) * 100)


def smape(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    denom = (np.abs(y) + np.abs(yhat)) / 2 + 1e-9
    return float(np.mean(np.abs(y - yhat) / denom) * 100)


def r2(y, yhat) -> float:
    return float(r2_score(y, yhat))


def interval_coverage(y, lo, hi) -> float:
    y, lo, hi = map(lambda a: np.asarray(a, float), (y, lo, hi))
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo, hi) -> float:
    return float(np.mean(np.asarray(hi, float) - np.asarray(lo, float)))


def regression_report(y, yhat, lo=None, hi=None) -> dict:
    out = {"mae": mae(y, yhat), "rmse": rmse(y, yhat), "mape_pct": mape(y, yhat),
           "smape_pct": smape(y, yhat), "r2": r2(y, yhat)}
    if lo is not None and hi is not None:
        out["coverage_90"] = interval_coverage(y, lo, hi)
        out["interval_width"] = interval_width(lo, hi)
    return {k: round(v, 4) for k, v in out.items()}


# --------------------------------------------------------------------------------------
# classification
# --------------------------------------------------------------------------------------
def classification_report(y, prob, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    pred = (prob >= threshold).astype(int)
    precision, recall, f1 = _safe_prf(y, pred)
    out = {
        "roc_auc": float(roc_auc_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y, prob)) if len(np.unique(y)) > 1 else float("nan"),
        "brier": float(brier_score_loss(y, prob)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "positive_rate": float(y.mean()),
    }
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in out.items()}


def calibration_bins(y, prob, n_bins: int = 10) -> list[dict]:
    """Reliability diagram data (are predicted probabilities honest?)."""
    y = np.asarray(y).astype(int)
    prob = np.asarray(prob, float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (prob >= bins[i]) & (prob < bins[i + 1]) if i < n_bins - 1 else (prob >= bins[i])
        if m.sum() == 0:
            continue
        out.append({"bin_lo": round(float(bins[i]), 2), "bin_hi": round(float(bins[i + 1]), 2),
                    "n": int(m.sum()), "mean_pred": round(float(prob[m].mean()), 4),
                    "observed": round(float(y[m].mean()), 4)})
    return out


# --------------------------------------------------------------------------------------
# anomaly detection
# --------------------------------------------------------------------------------------
def precision_at_k(y, score, k: int) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    k = min(k, len(y))
    idx = np.argsort(-score)[:k]
    return float(y[idx].sum() / k)


def recall_at_k(y, score, k: int) -> float:
    y = np.asarray(y).astype(int)
    score = np.asarray(score, float)
    total = y.sum()
    if total == 0:
        return float("nan")
    k = min(k, len(y))
    idx = np.argsort(-score)[:k]
    return float(y[idx].sum() / total)


def anomaly_report(y, score, ks=(100, 500, 1000)) -> dict:
    out = {}
    for k in ks:
        if k <= len(y):
            out[f"precision@{k}"] = round(precision_at_k(y, score, k), 4)
            out[f"recall@{k}"] = round(recall_at_k(y, score, k), 4)
    out["roc_auc"] = round(float(roc_auc_score(y, score)), 4) if len(np.unique(y)) > 1 else None
    out["pr_auc"] = round(float(average_precision_score(y, score)), 4) if len(np.unique(y)) > 1 else None
    return out


# --------------------------------------------------------------------------------------
# drift
# --------------------------------------------------------------------------------------
def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index — the classic production drift metric."""
    expected = np.asarray(expected, float)
    actual = np.asarray(actual, float)
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    if expected.size == 0 or actual.size == 0:
        return float("nan")
    qs = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(qs) < 3:
        return 0.0
    qs[0], qs[-1] = -np.inf, np.inf
    e = np.histogram(expected, qs)[0] / len(expected)
    a = np.histogram(actual, qs)[0] / len(actual)
    e = np.clip(e, 1e-6, None)
    a = np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


PSI_THRESHOLDS = {"no_shift": 0.10, "moderate": 0.25, "major": 999.0}


def psi_status(value: float) -> str:
    if value < PSI_THRESHOLDS["no_shift"]:
        return "no_shift"
    if value < PSI_THRESHOLDS["moderate"]:
        return "moderate_shift"
    return "major_shift"
