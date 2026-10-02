"""
Supervised labels for the risk model.

Definition (deliberately simple and defensible):

    A user is `stressed` if, within the 6 months AFTER the feature cut-off, any
    of these happen:
      * the running cash balance goes negative (they could not pay obligations)
      * net cash flow is negative in >= 3 of the 6 months
      * income collapses > 40% below the trailing median while liquid cover < 1 month

Features are computed strictly BEFORE the outcome window, so there is no leakage.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from common.logging_utils import get_logger
from common.utils import safe_div

log = get_logger("labels")

OUTCOME_MONTHS = 6


def outcome_labels(panel: pd.DataFrame, cutoff: date, outcome_months: int = OUTCOME_MONTHS) -> pd.DataFrame:
    p = panel[panel["year_month"] > pd.Timestamp(cutoff)].copy()
    p = p.sort_values(["user_id", "year_month"])
    window = p.groupby("user_id").head(outcome_months)

    agg = window.groupby("user_id").agg(
        min_cash=("cash_balance", "min"),
        neg_months=("net_cash_flow", lambda s: int((s < 0).sum())),
        months=("net_cash_flow", "size"),
        mean_ncf=("net_cash_flow", "mean"),
        min_income=("income", "min"),
        mean_expense=("expense", "mean"),
        min_liquid=("liquid_assets", "min"),
    )
    trailing = (panel[panel["year_month"] <= pd.Timestamp(cutoff)]
                .groupby("user_id")["income"].median().rename("trailing_income"))
    agg = agg.join(trailing)

    # "materially" overdrawn: more than 20% of a month's expenses below zero
    stress = (
        (agg["min_cash"] < -0.2 * agg["mean_expense"]) |
        (agg["neg_months"] >= 3) |
        ((agg["min_income"] < 0.6 * agg["trailing_income"]) &
         (agg["min_liquid"] < agg["mean_expense"]))
    ).astype(int)
    out = pd.DataFrame({"user_id": agg.index, "stress_label": stress.to_numpy()})
    out["min_cash"] = agg["min_cash"].round(0).to_numpy()
    out["neg_months"] = agg["neg_months"].to_numpy()
    out["outcome_months"] = agg["months"].to_numpy()
    return out


def build_training_dataset(panel: pd.DataFrame, feats_at_cutoff: pd.DataFrame, cutoff: date,
                           outcome_months: int = OUTCOME_MONTHS) -> pd.DataFrame:
    lab = outcome_labels(panel, cutoff, outcome_months)
    ds = feats_at_cutoff.merge(lab, on="user_id", how="inner")
    log.info("training dataset @ %s: %s rows, stress rate %.3f",
             cutoff, f"{len(ds):,}", ds["stress_label"].mean())
    return ds
