"""
Supervised dataset for the forecasting models.

One row per (user, month) with only *information available at that month*:
lags, rolling statistics, calendar seasonality and behavioural ratios. The
target is the mean value over the next `h` months (direct multi-horizon
strategy, which avoids the error accumulation of recursive forecasting).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger

log = get_logger("forecast.dataset")

LAG_COLS = {
    "expense": ["expense", "income", "net_cash_flow", "emi", "discretionary_spend", "tx_count"],
    "income": ["income", "expense", "net_cash_flow", "tx_count"],
}


@dataclass
class ForecastDataset:
    X: pd.DataFrame
    y: dict[int, np.ndarray]     # horizon -> target (NaN where look-ahead is incomplete)
    groups: np.ndarray           # user_id per row (for grouped CV)
    dates: pd.Series
    feature_names: list
    train_mask: dict = field(default_factory=dict)   # horizon -> boolean mask of rows with a target


def build_forecast_dataset(panel: pd.DataFrame, target_col: str = "expense",
                           horizons: tuple[int, ...] = (1, 3, 6, 12),
                           max_lag: int = 12) -> ForecastDataset:
    p = panel.sort_values(["user_id", "year_month"]).copy()
    p["month"] = p["year_month"].dt.month
    p["t"] = (p["year_month"].dt.year - p["year_month"].dt.year.min()) * 12 + p["year_month"].dt.month

    g = p.groupby("user_id", sort=False)
    feats = {}
    base_cols = LAG_COLS.get(target_col, LAG_COLS["expense"])
    for c in base_cols:
        for lag in (1, 2, 3, 6, 12):
            feats[f"{c}_lag{lag}"] = g[c].shift(lag)
        for w in (3, 6, 12):
            roll = g[c].rolling(w, min_periods=1)
            feats[f"{c}_rmean{w}"] = roll.mean().reset_index(level=0, drop=True)
            feats[f"{c}_rstd{w}"] = roll.std().reset_index(level=0, drop=True)

    X = pd.DataFrame(feats, index=p.index)
    X["month_sin"] = np.sin(2 * np.pi * p["month"] / 12)
    X["month_cos"] = np.cos(2 * np.pi * p["month"] / 12)
    X["t"] = p["t"]
    X["user_income_level"] = g["income"].transform("mean")
    X["user_expense_level"] = g["expense"].transform("mean")
    X["expense_to_income"] = X["user_expense_level"] / (X["user_income_level"] + 1)
    X["emi_ratio"] = p["emi"] / (p["income"] + 1)
    X["discretionary_share"] = p["discretionary_spend"] / (p["expense"] + 1)
    for c in ("Housing", "Food", "Transport", "Utilities"):
        X[f"share_{c}"] = p[f"exp_{c}"] / (p["expense"] + 1)

    # targets: mean of the next h months (only rows with full look-ahead are kept later)
    y = {}
    valid = pd.Series(True, index=p.index)
    for h in horizons:
        fwd = g[target_col].shift(-1).rolling(h, min_periods=h).mean().reset_index(level=0, drop=True)
        y[h] = fwd.to_numpy()
        valid &= pd.Series(~np.isnan(y[h]), index=p.index)

    X = X.replace([np.inf, -np.inf], np.nan)
    mask = X.notna().all(axis=1).to_numpy()
    X = X[mask].reset_index(drop=True)
    dates = p.loc[mask, "year_month"].reset_index(drop=True)
    groups = p.loc[mask, "user_id"].to_numpy()
    valid_np = valid.to_numpy()[mask]
    y = {h: np.asarray(v)[mask] for h, v in y.items()}
    train_mask = {h: (valid_np & ~np.isnan(y[h]) & (p.loc[mask, "user_id"].to_numpy() ==
                                                    p.loc[mask, "user_id"].to_numpy())) for h in horizons}
    train_mask = {h: ~np.isnan(y[h]) & valid_np for h in horizons}
    log.info("forecast dataset target=%s rows=%s features=%s trainable=%s",
             target_col, f"{len(X):,}", X.shape[1], f"{int(train_mask[horizons[0]].sum()):,}")
    return ForecastDataset(X=X, y=y, groups=groups, dates=dates,
                           feature_names=list(X.columns), train_mask=train_mask)


def seasonal_naive_baseline(panel: pd.DataFrame, target_col: str, horizons: tuple[int, ...]) -> dict[int, np.ndarray]:
    """Simple, honest baseline: last observed value repeated (with 12-month drift)."""
    p = panel.sort_values(["user_id", "year_month"])
    last = p.groupby("user_id")[target_col].apply(lambda s: s.iloc[-1]).to_numpy()
    return {h: np.repeat(last[:, None], 1, axis=1).ravel() for h in horizons}
