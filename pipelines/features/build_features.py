"""
Feature store + Financial DNA engine.

Two outputs:
  1. `financial_features` — ~45 engineered features per user at a given as-of date
     (cash-flow, spending, debt, savings and behavioural blocks).
  2. `financial_dna` — a proprietary, interpretable 9-dimension profile scaled
     0–100, plus a composite financial-health score and behavioural segment.

The DNA vector is deliberately *rule-based* (not a black box): every dimension
maps back to an explicit financial ratio, which is what makes the explainability
layer credible.
"""
from __future__ import annotations

import json
from datetime import date

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from common.utils import minmax_scale, safe_div

log = get_logger("features")

FEATURE_BLOCKS = {
    "cash_flow": ["monthly_income", "monthly_expense", "net_cash_flow", "cash_flow_volatility",
                  "income_volatility", "income_trend", "expense_volatility"],
    "spending": ["discretionary_ratio", "essential_ratio", "category_entropy", "category_hhi",
                 "spending_growth", "top_category_share"],
    "debt": ["debt_to_income", "debt_service_ratio", "loan_utilization", "has_debt", "n_loans"],
    "savings": ["savings_rate", "emergency_fund_months", "goal_progress", "monthly_surplus",
                "investment_ratio"],
    "behavioural": ["weekend_spending_ratio", "impulse_spending_score", "subscription_growth",
                    "spending_consistency", "latenight_ratio", "anomaly_rate", "tx_per_month"],
    "balance_sheet": ["liquid_assets", "cash_balance", "invest_value", "debt_balance", "net_worth",
                      "liquidity_ratio", "cashflow_coverage_months"],
}


def _entropy(shares: np.ndarray) -> np.ndarray:
    p = shares / np.clip(shares.sum(axis=1, keepdims=True), 1e-12, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        return -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1) / np.log(p.shape[1])


def _hhi(shares: np.ndarray) -> np.ndarray:
    p = shares / np.clip(shares.sum(axis=1, keepdims=True), 1e-12, None)
    return (p ** 2).sum(axis=1)


def _slope(y: np.ndarray) -> np.ndarray:
    """Per-row OLS slope against a fixed 0..n-1 x-axis (vectorised)."""
    n = y.shape[1]
    x = np.arange(n, dtype=float)
    x = x - x.mean()
    return (y * x).sum(axis=1) / (x ** 2).sum()


# --------------------------------------------------------------------------------------
# 1. engineered features
# --------------------------------------------------------------------------------------
def compute_features(panel: pd.DataFrame, users: pd.DataFrame, goals: pd.DataFrame,
                     debts: pd.DataFrame, as_of: date, window: int = 12) -> pd.DataFrame:
    p = panel[panel["year_month"] <= pd.Timestamp(as_of)].copy()
    p = p.sort_values(["user_id", "year_month"])
    last_n = p.groupby("user_id").tail(window)

    def agg(col, how="mean"):
        return last_n.groupby("user_id")[col].agg(how)

    n_months = last_n.groupby("user_id").size()
    mats = {}
    for col in ["income", "expense", "emi", "net_cash_flow", "tx_count", "tx_max", "tx_mean",
                "weekend_spend", "latenight_spend", "discretionary_spend", "anomaly_count",
                "cash_balance", "liquid_assets", "invest_value", "debt_balance", "invest_flow"]:
        wide = p.groupby("user_id")[col].apply(lambda s: s.to_numpy(dtype="float64"))
        mats[col] = np.vstack([np.pad(a[-window:], (window - len(a[-window:]), 0), constant_values=np.nan)
                               for a in wide])

    latest = p.groupby("user_id").tail(1).set_index("user_id")
    inc_m, exp_m = mats["income"], mats["expense"]
    ncf_m, emi_m = mats["net_cash_flow"], mats["emi"]

    with np.errstate(invalid="ignore"):
        income_mean = np.nanmean(inc_m, axis=1)
        expense_mean = np.nanmean(exp_m, axis=1)
        income_std = np.nanstd(inc_m, axis=1)
        expense_std = np.nanstd(exp_m, axis=1)
        ncf_std = np.nanstd(ncf_m, axis=1)

    cat_cols = [f"exp_{c}" for c in settings.EXPENSE_CATEGORIES]
    cat_totals = np.nan_to_num(last_n.groupby("user_id")[cat_cols].sum().to_numpy())
    essential_idx = [settings.EXPENSE_CATEGORIES.index(c) for c in settings.ESSENTIAL_CATEGORIES]
    disc_idx = [settings.EXPENSE_CATEGORIES.index(c) for c in settings.DISCRETIONARY_CATEGORIES]

    total_cat = cat_totals.sum(axis=1)
    essential_ratio = cat_totals[:, essential_idx].sum(axis=1) / np.clip(total_cat, 1e-9, None)
    disc_ratio = cat_totals[:, disc_idx].sum(axis=1) / np.clip(total_cat, 1e-9, None)

    # spending growth: last 3 months vs the 3 before
    last3 = np.nanmean(exp_m[:, -3:], axis=1)
    prev3 = np.nanmean(exp_m[:, -6:-3], axis=1)
    spending_growth = safe_div(last3 - prev3, prev3, 0.0)

    sub_series = last_n.groupby("user_id")["exp_Subscriptions"].mean()
    sub_first = p.groupby("user_id")["exp_Subscriptions"].apply(lambda s: s.iloc[:3].mean())
    subscription_growth = safe_div(sub_series.to_numpy() - sub_first.to_numpy(),
                                   np.abs(sub_first.to_numpy()), 0.0)

    tx_max_mean = np.nanmean(mats["tx_max"], axis=1)
    impulse = 0.55 * minmax_scale(pd.Series(safe_div(tx_max_mean, np.nanmean(mats["tx_mean"], axis=1), 0))) + \
              0.45 * minmax_scale(pd.Series(safe_div(np.nanmean(mats["latenight_spend"], axis=1), expense_mean, 0)))

    feats = pd.DataFrame(index=last_n.groupby("user_id").size().index)
    feats.index.name = "user_id"
    feats["as_of"] = pd.Timestamp(as_of)
    feats["months_observed"] = n_months

    # ---- cash flow ---------------------------------------------------------------
    feats["monthly_income"] = np.nanmean(inc_m[:, -6:], axis=1)
    feats["monthly_expense"] = expense_mean
    feats["monthly_emi"] = np.nanmean(emi_m, axis=1)
    feats["net_cash_flow"] = np.nanmean(ncf_m, axis=1)
    feats["cash_flow_volatility"] = safe_div(ncf_std, np.abs(np.nanmean(ncf_m, axis=1)), 0.0)
    feats["income_volatility"] = safe_div(income_std, income_mean, 0.0)
    feats["expense_volatility"] = safe_div(expense_std, expense_mean, 0.0)
    feats["income_trend"] = _slope(inc_m) / np.clip(income_mean, 1, None)

    # ---- spending ----------------------------------------------------------------
    feats["discretionary_ratio"] = disc_ratio
    feats["essential_ratio"] = essential_ratio
    feats["category_entropy"] = _entropy(cat_totals)
    feats["category_hhi"] = _hhi(cat_totals)
    feats["top_category_share"] = cat_totals.max(axis=1) / np.clip(total_cat, 1e-9, None)
    feats["spending_growth"] = spending_growth
    feats["spending_consistency"] = 1.0 / (1.0 + safe_div(expense_std, expense_mean, 0.0))

    # ---- debt --------------------------------------------------------------------
    dbt = debts.groupby("user_id").agg(
        principal=("principal", "sum"), remaining=("remaining_balance", "sum"),
        payment=("monthly_payment", "sum"), n_loans=("loan_id", "size"),
        rate=("interest_rate", "mean")).reindex(feats.index).fillna(0)
    feats["debt_balance"] = dbt["remaining"]
    feats["debt_to_income"] = safe_div(dbt["remaining"], feats["monthly_income"] * 12, 0.0)
    feats["debt_service_ratio"] = safe_div(dbt["payment"], feats["monthly_income"], 0.0)
    feats["loan_utilization"] = safe_div(dbt["remaining"], dbt["principal"], 0.0)
    feats["avg_interest_rate"] = dbt["rate"]
    feats["n_loans"] = dbt["n_loans"].astype(int)
    feats["has_debt"] = (dbt["remaining"] > 0).astype(int)

    # ---- savings -----------------------------------------------------------------
    feats["savings_rate"] = safe_div(feats["net_cash_flow"], feats["monthly_income"], 0.0)
    feats["monthly_surplus"] = feats["net_cash_flow"]
    feats["cash_balance"] = latest["cash_balance"].reindex(feats.index).fillna(0)
    feats["liquid_assets"] = latest["liquid_assets"].reindex(feats.index).fillna(0)
    feats["invest_value"] = latest["invest_value"].reindex(feats.index).fillna(0)
    feats["emergency_fund_months"] = safe_div(feats["liquid_assets"], expense_mean, 0.0)
    feats["investment_ratio"] = safe_div(feats["invest_value"], feats["monthly_income"] * 12, 0.0)
    feats["net_worth"] = feats["cash_balance"] + feats["invest_value"] - feats["debt_balance"]
    feats["liquidity_ratio"] = feats["emergency_fund_months"]
    feats["cashflow_coverage_months"] = safe_div(feats["liquid_assets"],
                                                 (expense_mean + np.nanmean(emi_m, axis=1)), 0.0)

    # ---- goals -------------------------------------------------------------------
    g = goals.sort_values("priority").groupby("user_id").first().reindex(feats.index)
    feats["goal_target"] = g["target_amount"].fillna(0)
    feats["goal_current"] = g["current_amount"].fillna(0)
    feats["goal_progress"] = safe_div(g["current_amount"].fillna(0), g["target_amount"], 0.0)
    months_left = ((pd.to_datetime(g["target_date"]) - pd.Timestamp(as_of)).dt.days / 30.44).clip(lower=0)
    feats["goal_months_left"] = months_left.fillna(0)
    required = safe_div(g["target_amount"] - g["current_amount"], months_left.clip(lower=1), 0)
    feats["goal_required_monthly"] = np.where(required > 0, required, 0.0)
    feasibility = safe_div(feats["monthly_surplus"], feats["goal_required_monthly"], 0.0)
    feats["goal_feasibility"] = np.where(feats["goal_required_monthly"] <= 0, 1.5, feasibility)

    # ---- behavioural ---------------------------------------------------------------
    feats["weekend_spending_ratio"] = safe_div(np.nanmean(mats["weekend_spend"], axis=1), expense_mean, 0.0)
    feats["latenight_ratio"] = safe_div(np.nanmean(mats["latenight_spend"], axis=1), expense_mean, 0.0)
    feats["impulse_spending_score"] = impulse.to_numpy()
    feats["subscription_growth"] = subscription_growth
    feats["anomaly_rate"] = safe_div(np.nanmean(mats["anomaly_count"], axis=1),
                                     np.nanmean(mats["tx_count"], axis=1), 0.0)
    feats["tx_per_month"] = np.nanmean(mats["tx_count"], axis=1)

    feats = feats.reset_index()
    users_cols = ["user_id", "persona", "age", "city", "risk_preference", "household_size",
                  "income_type", "employment_type", "financial_goal", "monthly_income_base"]
    feats = feats.merge(users[users_cols], on="user_id", how="left")

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    log.info("features: %s users × %s columns @ %s", f"{len(feats):,}", feats.shape[1], as_of)
    return feats


# --------------------------------------------------------------------------------------
# 2. Financial DNA
# --------------------------------------------------------------------------------------
def _score_from_ratio(s: pd.Series, good: float, bad: float, invert: bool = False) -> pd.Series:
    """Map a ratio onto 0-100.

    invert=False -> 100 when the ratio is at least `good`, 0 at `bad` (higher is better)
    invert=True  -> 100 when the ratio is at most `good`, 0 at `bad` (lower is better)
    """
    s = pd.Series(s).astype(float)
    if good == bad:
        return pd.Series(50.0, index=s.index)
    lo, hi = (bad, good) if not invert else (good, bad)
    raw = ((s - lo) / (hi - lo) * 100).clip(0, 100)
    return (100 - raw) if invert else raw


def compute_dna(feats: pd.DataFrame) -> pd.DataFrame:
    f = feats
    dna = pd.DataFrame(index=f.index)
    dna["liquidity"] = _score_from_ratio(f["emergency_fund_months"], good=5.0, bad=0.0)
    dna["savings_discipline"] = _score_from_ratio(f["savings_rate"], good=0.30, bad=-0.10)
    dna["spending_stability"] = _score_from_ratio(f["expense_volatility"], good=0.05, bad=0.60, invert=True)
    dna["debt_resilience"] = (0.5 * _score_from_ratio(f["debt_service_ratio"], good=0.05, bad=0.60, invert=True) +
                              0.5 * _score_from_ratio(f["debt_to_income"], good=0.2, bad=4.0, invert=True))
    dna["income_stability"] = _score_from_ratio(f["income_volatility"], good=0.05, bad=0.55, invert=True)
    dna["investment_exposure"] = _score_from_ratio(f["investment_ratio"], good=0.75, bad=0.0)
    dna["goal_discipline"] = _score_from_ratio(f["goal_feasibility"], good=1.5, bad=0.0)
    dna["emergency_resilience"] = (0.6 * _score_from_ratio(f["cashflow_coverage_months"], good=5.0, bad=0.0) +
                                   0.4 * _score_from_ratio(f["net_cash_flow"] / (f["monthly_expense"] + 1),
                                                           good=0.5, bad=-0.2))
    # composite risk: the weighted absence of the protective factors
    dna["financial_risk"] = 100 - (
        0.20 * dna["liquidity"] + 0.18 * dna["debt_resilience"] + 0.16 * dna["savings_discipline"] +
        0.14 * dna["income_stability"] + 0.12 * dna["emergency_resilience"] +
        0.10 * dna["spending_stability"] + 0.10 * dna["goal_discipline"]
    )
    dna["financial_health"] = 100 - dna["financial_risk"]

    dna = dna.clip(0, 100).round(1)
    for c in dna.columns:
        feats[f"dna_{c}"] = dna[c]

    # legacy / presentation aliases
    feats["health_score"] = feats["dna_financial_health"]
    feats["risk_score_rule"] = feats["dna_financial_risk"]
    log.info("financial DNA computed for %s users (mean health %.1f)",
             f"{len(feats):,}", feats["dna_financial_health"].mean())
    return feats


DNA_DIMENSIONS = ("liquidity", "savings_discipline", "spending_stability", "debt_resilience",
                  "income_stability", "investment_exposure", "goal_discipline",
                  "emergency_resilience", "financial_risk")


# --------------------------------------------------------------------------------------
# 3. behavioural segmentation
# --------------------------------------------------------------------------------------
SEGMENT_NAMES = {
    "resilient_builder": "Resilient Builder",
    "comfortable_leveraged": "Comfortable but Leveraged",
    "high_earner_thin_buffer": "High Earner / Thin Buffer",
    "overextended": "Overextended",
    "fragile": "Financially Fragile",
    "aggressive_allocator": "Aggressive Allocator",
}


def segment(feats: pd.DataFrame, k: int = 6, seed: int = 42) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    cols = [f"dna_{d}" for d in DNA_DIMENSIONS if d != "financial_risk"]
    X = StandardScaler().fit_transform(feats[cols].to_numpy())
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(X)
    feats["segment_id"] = km.labels_.astype(int)

    centres = pd.DataFrame(km.cluster_centers_, columns=cols)
    names = {}
    prof = feats.groupby("segment_id")[["monthly_income", "emergency_fund_months", "savings_rate",
                                        "debt_service_ratio", "dna_investment_exposure",
                                        "dna_financial_health"]].mean()
    med_income = prof["monthly_income"].median()

    def preferred(cid) -> str:
        row = prof.loc[cid]
        if row["dna_financial_health"] < 40:
            return "fragile"
        if row["debt_service_ratio"] > 0.35 or row["debt_service_ratio"] > prof["debt_service_ratio"].quantile(0.8):
            return "overextended"
        if row["dna_investment_exposure"] > 70 and row["emergency_fund_months"] > 3:
            return "aggressive_allocator"
        if row["emergency_fund_months"] < 2.0 and row["monthly_income"] > med_income:
            return "high_earner_thin_buffer"
        if row["emergency_fund_months"] > 4 and row["savings_rate"] > 0.15:
            return "resilient_builder"
        return "comfortable_leveraged"

    pool = ["resilient_builder", "aggressive_allocator", "high_earner_thin_buffer",
            "comfortable_leveraged", "overextended", "fragile"]
    used: set[str] = set()
    for cid in prof["dna_financial_health"].sort_values(ascending=False).index:
        pref = preferred(cid)
        if pref not in used:
            names[cid] = pref
            used.add(pref)
        else:
            free = [n for n in pool if n not in used]
            names[cid] = free[0] if free else f"{pref}_b"
            used.add(names[cid])
    feats["segment"] = feats["segment_id"].map(names)
    feats["segment_label"] = feats["segment"].map(SEGMENT_NAMES)
    log.info("segments: %s", feats["segment_label"].value_counts().to_dict())
    return feats
