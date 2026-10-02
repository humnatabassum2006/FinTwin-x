"""
Scenario engine — turns "what if …?" into numbers.

A scenario is a declarative parameter delta. The engine resolves presets, merges
user overrides, runs the Monte-Carlo engine and returns a comparison table that
the dashboard and the AI agents both consume. The LLM never invents these
numbers: it calls this module.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Any, Iterable

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from common import settings
from common.logging_utils import get_logger
from data_generator.scenarios import TEMPLATES
from simulation.monte_carlo import (ScenarioParams, SimConfig, SimulationResult,
                                    UserState, run_simulation)

log = get_logger("scenarios")


class ScenarioSpec(BaseModel):
    """API-facing scenario contract (validated, serialised, LLM-tool-callable)."""
    name: str = "custom"
    label: str = "Custom scenario"
    preset: str | None = None
    horizon_months: int = Field(default=36, ge=1, le=settings.MAX_SIM_HORIZON)
    n_paths: int = Field(default=5000, ge=500, le=100_000)

    income_change_pct: float = 0.0
    income_shock_pct: float = 0.0
    shock_duration_months: int = 0
    shock_start_month: int = 1
    recovery_factor: float = 1.0
    extra_monthly_income: float = 0.0

    expense_change_pct: float = 0.0
    inflation_rate: float | None = None
    lump_sum_expense: float = 0.0
    lump_sum_month: int = 1

    monthly_investment: float | None = None
    invest_return_mu: float | None = None

    new_debt_principal: float = 0.0
    new_debt_rate: float = 0.20
    new_debt_term_months: int = 60
    new_debt_start_month: int = 1
    debt_payment_multiplier: float = 1.0

    goal_target: float | None = None
    goal_current: float | None = None
    goal_month: int | None = None

    def to_params(self) -> ScenarioParams:
        d = self.model_dump(exclude={"preset", "n_paths"})
        d = {k: v for k, v in d.items() if v is not None}
        return ScenarioParams(**d)


def resolve_preset(preset: str, overrides: dict[str, Any] | None = None) -> ScenarioSpec:
    """Map a preset key (job_loss, car_purchase, …) onto a validated spec."""
    if preset not in TEMPLATES:
        raise KeyError(f"unknown preset '{preset}'. Available: {list(TEMPLATES)}")
    t = TEMPLATES[preset]
    payload = {"name": preset, "label": t.label, "preset": preset, **t.params}
    payload.update(overrides or {})
    return ScenarioSpec(**payload)


def preset_specs(user_row=None) -> list[ScenarioSpec]:
    specs = []
    for key, t in TEMPLATES.items():
        params = dict(t.params)
        if key == "invest_more" and user_row is not None:
            params["monthly_investment"] = float(min(40_000, max(user_row.get("monthly_surplus", 20_000), 20_000)))
        specs.append(ScenarioSpec(name=key, label=t.label, preset=key, **params))
    return specs


def run_scenario(user_row, spec: ScenarioSpec, cfg: SimConfig | None = None) -> dict:
    state = UserState.from_features(user_row)
    cfg = cfg or SimConfig(n_paths=spec.n_paths, horizon_months=spec.horizon_months)
    params = spec.to_params()
    if params.goal_target is None:
        params.goal_target = state.goal_target
    if params.goal_current is None:
        params.goal_current = state.goal_current
    if params.goal_month is None:
        params.goal_month = state.goal_month
    res = run_simulation(state, params, cfg)
    return {
        "name": res.name,
        "label": res.label,
        "summary": res.summary,
        "bands": res.bands,
        "histogram": res.histogram,
        "assumptions": res.assumptions,
        "changes": params.describe(),
    }


COMPARE_FIELDS = [
    ("goal_probability", "Goal probability", "pct"),
    ("stress_probability", "Stress probability", "pct"),
    ("expected_final_net_worth", "Expected net worth", "money"),
    ("p5_final_net_worth", "Downside (5th pct)", "money"),
    ("expected_final_liquid", "Liquid assets", "money"),
    ("median_months_of_cover_at_end", "Months of cover", "num"),
    ("expected_final_debt", "Debt at horizon", "money"),
]


def compare_scenarios(user_row, specs: Iterable[ScenarioSpec], cfg: SimConfig | None = None) -> dict:
    """Run several scenarios and return a decision-ready comparison table."""
    results = [run_scenario(user_row, s, cfg) for s in specs]
    baseline = results[0]
    rows = []
    for key, label, kind in COMPARE_FIELDS:
        base_val = baseline["summary"].get(key)
        row = {"metric": key, "label": label, "kind": kind,
               "values": [r["summary"].get(key) for r in results],
               "baseline": base_val}
        row["deltas"] = [None if (v is None or base_val is None) else round(v - base_val, 4)
                         for v in row["values"]]
        rows.append(row)
    return {
        "scenarios": [{"name": r["name"], "label": r["label"], "changes": r["changes"],
                       "summary": r["summary"], "assumptions": r["assumptions"]} for r in results],
        "table": rows,
        "curves": {r["name"]: r["bands"] for r in results},
        "baseline_name": baseline["name"],
    }


# --------------------------------------------------------------------------------------
# decision optimiser — "what is the safest way to reach my goal?"
# --------------------------------------------------------------------------------------
def optimise_decision(user_row, base_spec: ScenarioSpec, target_metric: str = "goal_probability",
                      target_value: float = 0.75, cfg: SimConfig | None = None) -> dict:
    """Search a small, human-meaningful action space and rank the options."""
    state = UserState.from_features(user_row)
    cfg = cfg or SimConfig(n_paths=2500, horizon_months=base_spec.horizon_months)
    baseline = run_scenario(user_row, base_spec, cfg)

    options: list[dict] = []

    def add(key, label, overrides, description):
        spec = base_spec.model_copy(update=overrides)
        res = run_scenario(user_row, spec, cfg)
        options.append({
            "key": key, "label": label, "description": description,
            "summary": res["summary"], "overrides": overrides,
        })

    start_month = int(base_spec.new_debt_start_month or 1)
    has_decision = any([base_spec.new_debt_principal, base_spec.lump_sum_expense,
                        base_spec.income_change_pct, base_spec.expense_change_pct,
                        base_spec.income_shock_pct])

    if not has_decision:
        # no specific decision in the question → search goal-achievement levers
        state0 = UserState.from_features(user_row)
        for extra in (10_000, 25_000, 50_000):
            add(f"contrib_plus_{extra}", f"Save Rs {extra:,} more per month",
                {"monthly_goal_contribution": extra, "monthly_investment": extra},
                f"Increase the monthly contribution by Rs {extra:,}.")
        add("extend_12m", "Extend the deadline by 12 months",
            {"goal_month": int(state0.goal_month) + 12,
             "horizon_months": max(int(base_spec.horizon_months), int(state0.goal_month) + 12)},
            "Give the goal one more year.")
        add("cut_spend_10", "Cut expenses 10% and redirect the surplus",
            {"expense_change_pct": -0.10},
            "Free up cash flow and save the difference.")
        add("income_up_10", "Increase income 10% and save the difference",
            {"income_change_pct": 0.10},
            "A raise or side income dedicated to the goal.")
        return {
            "target_metric": target_metric, "target_value": target_value,
            "baseline": {"label": baseline["label"], "summary": baseline["summary"]},
            "options": sorted(options, key=lambda o: -(o["summary"].get(target_metric) or 0)),
            "recommended": None, "meets_target": False,
            "note": "No specific purchase was detected, so the search covered contribution, "
                    "deadline, spending and income levers.",
        }

    add("buy_now", "Buy now", {}, "Take the decision immediately.")
    for delay in (6, 12, 18, 24):
        add(f"delay_{delay}m", f"Delay by {delay} months",
            {"new_debt_start_month": start_month + delay, "lump_sum_month": start_month + delay},
            f"Postpone the purchase by {delay} months and keep saving.")
    add("smaller_ticket", "Smaller ticket (60%)",
        {"new_debt_principal": round((base_spec.new_debt_principal or 0) * 0.6, -3),
         "lump_sum_expense": round((base_spec.lump_sum_expense or 0) * 0.6, -3)},
        "Choose a cheaper variant of the same purchase.")
    add("longer_tenor", "Stretch the loan to 84 months",
        {"new_debt_term_months": 84},
        "Lower EMI, more total interest.")
    add("income_boost_15", "Increase income 15%",
        {"income_change_pct": 0.15},
        "Side income / raise of 15% before committing.")
    add("cut_spend_15", "Cut discretionary spending 15%",
        {"expense_change_pct": -0.15},
        "Free up cash flow before committing.")

    scored = sorted(options, key=lambda o: -(o["summary"].get(target_metric) or 0))
    meets = [o for o in scored if (o["summary"].get(target_metric) or 0) >= target_value]
    return {
        "target_metric": target_metric,
        "target_value": target_value,
        "baseline": {"label": baseline["label"], "summary": baseline["summary"]},
        "options": scored,
        "recommended": meets[-1] if meets else scored[0],
        "meets_target": bool(meets),
        "note": "Options are ranked by the simulated metric; 'recommended' is the least "
                "aggressive option that still reaches the target.",
    }
