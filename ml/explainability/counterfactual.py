"""
Counterfactual engine — "what can I change?" (not just "why am I risky?").

Interventions are applied to a small set of *core state variables* and every
derived feature is recomputed with the same formulas the feature store uses, so
the counterfactual risk score is financially consistent — we never hand the
model an impossible feature combination.

Optionally each intervention is also re-simulated through the Monte-Carlo
engine, which gives a second, probabilistic opinion.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.logging_utils import get_logger

log = get_logger("counterfactual")

CORE_STATE = ["monthly_income", "monthly_expense", "monthly_emi", "cash_balance",
              "liquid_assets", "invest_value", "debt_balance", "goal_target",
              "goal_current", "goal_months_left", "net_cash_flow", "goal_required_monthly"]


@dataclass
class Intervention:
    key: str
    label: str
    detail: str
    apply: callable
    effort: str = "medium"        # easy | medium | hard


def derive(state: dict, row: pd.Series) -> dict:
    """Recompute all ratio features from a modified core state."""
    income = max(state["monthly_income"], 1.0)
    expense = max(state["monthly_expense"], 1.0)
    emi = max(state["monthly_emi"], 0.0)
    liquid = max(state["liquid_assets"], 0.0)
    invest = max(state["invest_value"], 0.0)
    debt = max(state["debt_balance"], 0.0)
    ncf = income - expense - emi

    out = dict(state)
    out.update({
        "net_cash_flow": ncf,
        "monthly_surplus": ncf,
        "savings_rate": ncf / income,
        "debt_service_ratio": emi / income,
        "debt_to_income": debt / (income * 12),
        "emergency_fund_months": liquid / expense,
        "liquidity_ratio": liquid / expense,
        "cashflow_coverage_months": liquid / (expense + emi),
        "investment_ratio": invest / (income * 12),
        "net_worth": state["cash_balance"] + invest - debt,
        "goal_required_monthly": max(state["goal_target"] - state["goal_current"], 0) /
                                 max(state["goal_months_left"], 1),
        "goal_progress": state["goal_current"] / max(state["goal_target"], 1),
        "goal_feasibility": (ncf / (max(state["goal_target"] - state["goal_current"], 0) /
                                    max(state["goal_months_left"], 1)))
        if state["goal_months_left"] > 0 else 1.5,
    })
    # Financial DNA recomputed with the same rules as the feature store
    from pipelines.features.build_features import _score_from_ratio
    dna = {
        "dna_liquidity": float(_score_from_ratio(pd.Series([out["emergency_fund_months"]]), 5.0, 0.0)[0]),
        "dna_savings_discipline": float(_score_from_ratio(pd.Series([out["savings_rate"]]), 0.30, -0.10)[0]),
        "dna_debt_resilience": 0.5 * float(_score_from_ratio(pd.Series([out["debt_service_ratio"]]), 0.05, 0.60, True)[0]) +
                               0.5 * float(_score_from_ratio(pd.Series([out["debt_to_income"]]), 0.2, 4.0, True)[0]),
        "dna_investment_exposure": float(_score_from_ratio(pd.Series([out["investment_ratio"]]), 0.75, 0.0)[0]),
        "dna_emergency_resilience": 0.6 * float(_score_from_ratio(pd.Series([out["cashflow_coverage_months"]]), 5.0, 0.0)[0]) +
                                    0.4 * float(_score_from_ratio(pd.Series([ncf / expense]), 0.5, -0.2)[0]),
        "dna_goal_discipline": float(_score_from_ratio(pd.Series([out["goal_feasibility"]]), 1.0, 0.0)[0]),
    }
    out.update(dna)
    return out


def _state_from_row(row: pd.Series) -> dict:
    return {k: float(row.get(k, 0.0)) for k in CORE_STATE}


def intervention_catalogue(row: pd.Series) -> list[Intervention]:
    expense = float(row["monthly_expense"])
    income = float(row["monthly_income"])
    liquid = float(row["liquid_assets"])
    emi = float(row["monthly_emi"])

    def mk(key, label, detail, fn, effort):
        return Intervention(key, label, detail, fn, effort)

    cats = [
        mk("cut_discretionary_20", "Cut discretionary spending 20%",
           f"Reduce discretionary spend by ~Rs {expense * float(row.get('discretionary_ratio', 0.3)) * 0.2:,.0f}/month",
           lambda s: {**s, "monthly_expense": s["monthly_expense"] -
                      expense * float(row.get("discretionary_ratio", 0.3)) * 0.20}, "medium"),
        mk("cut_expenses_10", "Reduce total expenses 10%",
           f"Save Rs {expense * 0.10:,.0f}/month",
           lambda s: {**s, "monthly_expense": s["monthly_expense"] * 0.90}, "easy"),
        mk("raise_income_10", "Increase income 10%",
           f"Earn Rs {income * 0.10:,.0f}/month more",
           lambda s: {**s, "monthly_income": s["monthly_income"] * 1.10}, "hard"),
        mk("build_emergency_200k", "Add Rs 200k to emergency fund",
           "Liquid buffer +Rs 200,000",
           lambda s: {**s, "liquid_assets": s["liquid_assets"] + 200_000,
                      "cash_balance": s["cash_balance"] + 200_000}, "medium"),
        mk("build_emergency_500k", "Add Rs 500k to emergency fund",
           "Liquid buffer +Rs 500,000",
           lambda s: {**s, "liquid_assets": s["liquid_assets"] + 500_000,
                      "cash_balance": s["cash_balance"] + 500_000}, "hard"),
        mk("refinance_debt", "Refinance / reduce EMI 15%",
           f"EMI down Rs {emi * 0.15:,.0f}/month",
           lambda s: {**s, "monthly_emi": s["monthly_emi"] * 0.85}, "medium"),
        mk("clear_debt", "Clear all debt",
           "Use liquid assets to retire the loan book",
           lambda s: {**s, "monthly_emi": 0.0,
                      "liquid_assets": max(s["liquid_assets"] - s["debt_balance"], 0.0),
                      "cash_balance": max(s["cash_balance"] - s["debt_balance"], 0.0),
                      "debt_balance": 0.0}, "hard"),
        mk("grow_income_20", "Increase income 20%",
           f"Earn Rs {income * 0.20:,.0f}/month more",
           lambda s: {**s, "monthly_income": s["monthly_income"] * 1.20}, "hard"),
    ]
    if emi <= 0:
        cats = [c for c in cats if c.key not in ("refinance_debt", "clear_debt")]
    return cats


def counterfactuals(model, row: pd.Series, target_score: float | None = None,
                    simulator=None) -> dict:
    """Score the current state and every single-intervention counterfactual."""
    base_state = _state_from_row(row)
    current = float(model.predict_proba(pd.DataFrame([row])[model.features].astype(float))[0]) * 100

    out = []
    for iv in intervention_catalogue(row):
        new_state = iv.apply(base_state)
        new_feats = derive(new_state, row)
        new_row = pd.Series({**row.to_dict(), **new_feats})
        prob = float(model.predict_proba(pd.DataFrame([new_row])[model.features].astype(float))[0]) * 100
        item = {
            "key": iv.key,
            "label": iv.label,
            "detail": iv.detail,
            "effort": iv.effort,
            "new_score": round(prob, 1),
            "delta": round(prob - current, 1),
        }
        if simulator is not None:
            try:
                res = simulator(new_row)
                item["simulated_stress_probability"] = res
            except Exception:
                pass
        out.append(item)

    out = sorted(out, key=lambda d: d["delta"])
    return {
        "current_score": round(current, 1),
        "options": out,
        "best": out[0] if out else None,
        "note": "Scores are model estimates under ceteris-paribus assumptions; they are not guarantees.",
    }


def plan_to_target(model, row: pd.Series, target_score: float = 40.0) -> dict:
    """Greedy search: cheapest combination of interventions that reaches the target."""
    state = _state_from_row(row)
    current = float(model.predict_proba(pd.DataFrame([row])[model.features].astype(float))[0]) * 100
    chosen, steps = [], current
    catalogue = intervention_catalogue(row)
    for _ in range(4):
        best = None
        for iv in catalogue:
            if iv.key in [c["key"] for c in chosen]:
                continue
            trial = {**state}
            for c in chosen:
                trial = next(x for x in catalogue if x.key == c["key"]).apply(trial)
            trial = iv.apply(trial)
            feats = derive(trial, row)
            trial_row = pd.Series({**row.to_dict(), **feats})
            p = float(model.predict_proba(pd.DataFrame([trial_row])[model.features].astype(float))[0]) * 100
            if best is None or p < best[1]:
                best = (iv, p)
        if best is None or best[1] >= steps - 0.5:
            break
        chosen.append({"key": best[0].key, "label": best[0].label, "detail": best[0].detail,
                       "score_after": round(best[1], 1)})
        steps = best[1]
        if steps <= target_score:
            break
    return {
        "current_score": round(current, 1),
        "target_score": target_score,
        "achieved_score": round(steps, 1),
        "reached": bool(steps <= target_score),
        "plan": chosen,
    }
