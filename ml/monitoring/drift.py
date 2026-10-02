"""
Production monitoring: distribution drift between the training population and
the population being scored today.

Metric: Population Stability Index (PSI) per feature.
    PSI < 0.10  no material shift
    0.10–0.25   moderate shift (investigate)
    > 0.25      major shift (retrain)
"""
from __future__ import annotations

import json
import time

import pandas as pd

from common import settings
from common.logging_utils import get_logger
from ml.evaluation.metrics import psi, psi_status

log = get_logger("drift")

MONITORED = [
    "monthly_income", "monthly_expense", "net_cash_flow", "savings_rate",
    "expense_volatility", "income_volatility", "discretionary_ratio",
    "essential_ratio", "category_entropy", "spending_growth",
    "debt_service_ratio", "debt_to_income", "emergency_fund_months",
    "investment_ratio", "impulse_spending_score", "weekend_spending_ratio",
    "tx_per_month", "goal_feasibility",
]


def _load(reference: pd.DataFrame | None, current: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    ref = reference if reference is not None else pd.read_parquet(
        settings.PROCESSED_DIR / "training_dataset.parquet")
    cur = current if current is not None else pd.read_parquet(
        settings.PROCESSED_DIR / "financial_features.parquet")
    return ref, cur


def drift_report(reference: pd.DataFrame | None = None, current: pd.DataFrame | None = None) -> dict:
    ref, cur = _load(reference, current)
    feats = [f for f in MONITORED if f in ref.columns and f in cur.columns]
    rows = []
    for f in feats:
        v = psi(ref[f].to_numpy(dtype=float), cur[f].to_numpy(dtype=float))
        rows.append({"feature": f, "psi": round(v, 4), "status": psi_status(v),
                     "ref_mean": round(float(ref[f].mean()), 4),
                     "cur_mean": round(float(cur[f].mean()), 4),
                     "shift_pct": (round(float(cur[f].mean() / (ref[f].mean() + 1e-9) - 1) * 100, 1)
                                   if abs(float(ref[f].mean())) > 1e-9 else None)})
    rows = sorted(rows, key=lambda r: -r["psi"])
    counts = {s: sum(1 for r in rows if r["status"] == s) for s in
              ("no_shift", "moderate_shift", "major_shift")}
    overall = max((r["psi"] for r in rows), default=0.0)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reference": "training_dataset (cut-off 6 months before as-of)",
        "current": "financial_features (as-of)",
        "n_reference": int(len(ref)), "n_current": int(len(cur)),
        "overall_psi": round(overall, 4),
        "overall_status": psi_status(overall),
        "counts": counts,
        "features": rows,
        "action": ("retrain recommended" if counts["major_shift"] else
                   ("monitor closely" if counts["moderate_shift"] else "no action required")),
    }
    (settings.PROCESSED_DIR / "drift_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    return report


def inject_drift_demo(scale: float = 1.6, feature: str = "monthly_expense") -> dict:
    """Demonstrate the detector: inflate one feature and re-measure PSI."""
    ref, cur = _load(None, None)
    cur = cur.copy()
    cur[feature] = cur[feature] * scale
    v = psi(ref[feature].to_numpy(dtype=float), cur[feature].to_numpy(dtype=float))
    log.info("injected drift demo: %s x%.1f → PSI %.3f (%s)", feature, scale, v, psi_status(v))
    return {"feature": feature, "scale": scale, "psi": round(v, 4), "status": psi_status(v),
            "message": (f"Simulated {int((scale-1)*100)}% growth in {feature}: "
                        f"PSI = {v:.3f} ({psi_status(v)}) — "
                        f"{'retrain recommended' if v > 0.25 else 'monitor'}")}
