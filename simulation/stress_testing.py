"""
Stress testing — the regulator's view of a household balance sheet.

A battery of adverse scenarios is run against the same user so the dashboard can
answer "how bad can it get?" instead of only "what is the average?".
"""
from __future__ import annotations

import numpy as np

from common.logging_utils import get_logger
from simulation.monte_carlo import ScenarioParams, SimConfig, UserState, run_simulation

log = get_logger("stress")

SHOCKS = [
    ("Income −20% (permanent)", dict(income_change_pct=-0.20)),
    ("Income −50% for 6 months", dict(income_shock_pct=-0.50, shock_duration_months=6, recovery_factor=0.95)),
    ("Job loss (100% for 4 months)", dict(income_shock_pct=-1.0, shock_duration_months=4, recovery_factor=0.9)),
    ("Job loss (100% for 9 months)", dict(income_shock_pct=-1.0, shock_duration_months=9, recovery_factor=0.85)),
    ("Inflation +4pp", dict(inflation_rate=0.115)),
    ("Inflation +8pp", dict(inflation_rate=0.155)),
    ("Medical emergency Rs 800k", dict(lump_sum_expense=800_000, lump_sum_month=6)),
    ("EMI +25% (rate reset)", dict(debt_payment_multiplier=1.25)),
    ("Investment returns −50%", dict(invest_return_mu=0.03)),
    ("Combined severe", dict(income_shock_pct=-0.5, shock_duration_months=6, recovery_factor=0.85,
                             inflation_rate=0.13, lump_sum_expense=500_000, lump_sum_month=3)),
]


def stress_matrix(user_row, horizon: int = 36, n_paths: int = 4000) -> dict:
    state = UserState.from_features(user_row)
    cfg = SimConfig(n_paths=n_paths, horizon_months=horizon)
    base = run_simulation(state, ScenarioParams(name="baseline", label="Baseline",
                                                horizon_months=horizon), cfg)

    rows = []
    for label, params in SHOCKS:
        sc = ScenarioParams(name=label.lower().replace(" ", "_"), label=label,
                            horizon_months=horizon, **params)
        res = run_simulation(state, sc, cfg)
        s = res.summary
        rows.append({
            "scenario": label,
            "stress_probability": s["stress_probability"],
            "goal_probability": s["goal_probability"],
            "expected_final_net_worth": s["expected_final_net_worth"],
            "p5_final_net_worth": s["p5_final_net_worth"],
            "months_of_cover": s["median_months_of_cover_at_end"],
            "delta_vs_baseline_stress": round(s["stress_probability"] - base.summary["stress_probability"], 4),
            "delta_vs_baseline_goal": (None if s["goal_probability"] is None or
                                       base.summary["goal_probability"] is None else
                                       round(s["goal_probability"] - base.summary["goal_probability"], 4)),
            "severity": _severity(s["stress_probability"]),
        })
    return {
        "baseline": base.summary,
        "horizon_months": horizon,
        "n_paths": n_paths,
        "tests": sorted(rows, key=lambda r: -r["stress_probability"]),
    }


def _severity(p: float) -> str:
    if p >= 0.5:
        return "severe"
    if p >= 0.25:
        return "high"
    if p >= 0.10:
        return "moderate"
    return "low"


def resilience_curve(user_row, drops=(0.0, -0.2, -0.4, -0.6, -0.8, -1.0),
                     horizon: int = 36, n_paths: int = 3000) -> dict:
    """How long does the household last at each income level? (survival curve)."""
    state = UserState.from_features(user_row)
    cfg = SimConfig(n_paths=n_paths, horizon_months=horizon)
    out = []
    for d in drops:
        sc = ScenarioParams(name=f"drop_{int(-d*100)}", label=f"Income {d:.0%}",
                            income_change_pct=d, horizon_months=horizon)
        res = run_simulation(state, sc, cfg)
        out.append({
            "income_change_pct": d,
            "stress_probability": res.summary["stress_probability"],
            "median_months_of_cover_at_end": res.summary["median_months_of_cover_at_end"],
            "expected_final_liquid": res.summary["expected_final_liquid"],
            "worst_case_month": res.summary["worst_case_month"],
        })
    return {"horizon_months": horizon, "curve": out}
