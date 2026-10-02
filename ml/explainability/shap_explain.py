"""
Explainable AI — SHAP attributions for the risk score.

Every risk score returned by FinTwin-X can be decomposed into per-feature
contributions measured in *risk points* (e.g. "debt burden +22"), so the user
sees why the score is what it is — not just the number.

Primary path: exact TreeSHAP (works for XGBoost / LightGBM).
Fallback: a local linear surrogate fitted on perturbations around the instance,
so the dashboard never loses its "why" panel even if SHAP is unavailable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger

log = get_logger("explain")

HUMAN_LABELS = {
    "monthly_income": "Income level",
    "monthly_expense": "Expense level",
    "monthly_emi": "EMI burden",
    "net_cash_flow": "Net cash flow",
    "savings_rate": "Savings rate",
    "emergency_fund_months": "Emergency reserve",
    "cashflow_coverage_months": "Cash-flow coverage",
    "debt_service_ratio": "Debt service ratio",
    "debt_to_income": "Debt-to-income",
    "loan_utilization": "Loan utilisation",
    "liquidity_ratio": "Liquidity",
    "expense_volatility": "Spending volatility",
    "income_volatility": "Income volatility",
    "cash_flow_volatility": "Cash-flow volatility",
    "discretionary_ratio": "Discretionary share",
    "essential_ratio": "Essential share",
    "category_entropy": "Spending diversity",
    "spending_growth": "Spending growth",
    "impulse_spending_score": "Impulse spending",
    "subscription_growth": "Subscription creep",
    "weekend_spending_ratio": "Weekend spending",
    "anomaly_rate": "Unusual transactions",
    "investment_ratio": "Investment exposure",
    "goal_feasibility": "Goal feasibility",
    "goal_progress": "Goal progress",
    "dna_financial_risk": "Rule-based risk",
    "dna_liquidity": "Liquidity DNA",
    "dna_debt_resilience": "Debt resilience DNA",
    "dna_income_stability": "Income stability DNA",
    "dna_savings_discipline": "Savings discipline DNA",
    "dna_emergency_resilience": "Emergency resilience DNA",
    "dna_spending_stability": "Spending stability DNA",
    "dna_goal_discipline": "Goal discipline DNA",
    "dna_investment_exposure": "Investment exposure DNA",
    "tx_per_month": "Transaction frequency",
    "spending_consistency": "Spending consistency",
}


def load_background(path: Path | None = None, n: int = 300) -> pd.DataFrame | None:
    path = Path(path or settings.MODEL_DIR / "risk_background.parquet")
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.sample(min(n, len(df)), random_state=settings.RANDOM_SEED)


def explain(model, X_row: pd.DataFrame, background: pd.DataFrame | None = None) -> dict:
    """Return SHAP contributions in risk-score points for a single user."""
    feats = list(model.features)
    x = X_row[feats].astype(float)
    base_prob = float(model.predict_proba(background if background is not None else x)[0]) if background is None \
        else float(np.mean(model.predict_proba(background)))
    prob = float(model.predict_proba(x)[0])

    contribs = _tree_shap(model, x, feats)
    if contribs is None:
        contribs = _surrogate_shap(model, x, background, feats)

    total = sum(abs(v) for v in contribs.values()) or 1.0
    items = sorted(contribs.items(), key=lambda kv: -abs(kv[1]))
    return {
        "score": round(prob * 100, 1),
        "probability": round(prob, 4),
        "base_score": round(base_prob * 100, 1),
        "contributions": [
            {
                "feature": f,
                "label": HUMAN_LABELS.get(f, f.replace("_", " ").title()),
                "points": round(v, 2),
                "share_pct": round(abs(v) / total * 100, 1),
                "direction": "increases_risk" if v > 0 else "reduces_risk",
                "value": float(x[f].iloc[0]),
            }
            for f, v in items if abs(v) >= 0.15
        ][:12],
        "method": "tree_shap" if _tree_shap is not None and _HAS_SHAP else "linear_surrogate",
        "top_driver": HUMAN_LABELS.get(items[0][0], items[0][0]) if items else None,
    }


_HAS_SHAP = bool(settings.ENABLE_SHAP)
if _HAS_SHAP:
    try:  # pragma: no cover - environment dependent
        import shap  # noqa: F401
    except Exception:  # pragma: no cover
        _HAS_SHAP = False
        log.warning("shap not installed — falling back to local surrogate explanations")
else:
    log.info("SHAP disabled by FINTWIN_ENABLE_SHAP — using local surrogate explanations")


def _tree_shap(model, x: pd.DataFrame, feats: list):
    if not _HAS_SHAP:
        return None
    try:
        import shap
        booster = model.model if model.algorithm == "xgboost" else model.model
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(x)
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[..., 1]
        vals = sv[0] if sv.shape[0] == 1 else sv
        return {f: float(v) * 100 for f, v in zip(feats, vals)}
    except Exception as exc:
        log.warning("tree shap failed (%s) — using surrogate", exc)
        return None


def _surrogate_shap(model, x: pd.DataFrame, background: pd.DataFrame | None, feats: list,
                    n_samples: int = 512) -> dict:
    """Local linear surrogate: perturb around the instance, fit weighted ridge."""
    rng = np.random.default_rng(settings.RANDOM_SEED)
    if background is None or len(background) == 0:
        sigma = np.abs(x.to_numpy(dtype=float)).ravel() * 0.1 + 1e-6
        samples = x.to_numpy(dtype=float) + rng.normal(0, 1, size=(n_samples, len(feats))) * sigma
    else:
        bg = background[feats].astype(float).to_numpy()
        base = x.to_numpy(dtype=float).ravel()
        idx = rng.integers(0, len(bg), size=(n_samples, len(feats)))
        mask = rng.random((n_samples, len(feats))) < 0.3
        samples = np.tile(base, (n_samples, 1))
        for j in range(len(feats)):                 # swap in values column by column
            sel = mask[:, j]
            if sel.any():
                samples[sel, j] = bg[idx[sel, j], j]
    probs = model.predict_proba(pd.DataFrame(samples, columns=feats))

    Xc = samples - samples.mean(axis=0)
    yc = probs - probs.mean()
    denom = (Xc ** 2).sum(axis=0) + 1e-9
    beta = (Xc * yc[:, None]).sum(axis=0) / denom
    delta = (x.to_numpy(dtype=float).ravel() - samples.mean(axis=0))
    return {f: float(b * d) * 100 for f, b, d in zip(feats, beta, delta)}
