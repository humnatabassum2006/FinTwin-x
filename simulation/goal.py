"""Goal simulator: probability of success, required contribution and sensitivity."""
from __future__ import annotations

import numpy as np

from common.logging_utils import get_logger
from simulation.monte_carlo import (ScenarioParams, SimConfig, UserState,
                                    required_monthly_contribution, run_simulation)

log = get_logger("goal")


def goal_outlook(user_row, horizon: int | None = None, n_paths: int = 5000) -> dict:
    state = UserState.from_features(user_row)
    h = int(horizon or max(min(state.goal_month, settings_horizon(state)), 12))
    cfg = SimConfig(n_paths=n_paths, horizon_months=h)
    sc = ScenarioParams(name="goal_base", label="Current plan", horizon_months=h,
                        goal_target=state.goal_target, goal_current=state.goal_current,
                        goal_month=min(state.goal_month, h))
    res = run_simulation(state, sc, cfg)
    s = res.summary

    targets = {}
    for p in (0.50, 0.75, 0.90):
        sol = required_monthly_contribution(state, target_probability=p,
                                            cfg=SimConfig(n_paths=2000, horizon_months=h))
        targets[f"p{int(p*100)}"] = sol

    # sensitivity: what does each extra Rs 10k/month buy?
    curve = []
    current_contrib = float(user_row.get("monthly_surplus", 0) or 0)
    for extra in (0, 10_000, 25_000, 50_000, 100_000):
        sc2 = ScenarioParams(name=f"c{extra}", label=f"+Rs {extra:,}", horizon_months=h,
                             monthly_goal_contribution=max(current_contrib, 0) + extra,
                             goal_target=state.goal_target, goal_current=state.goal_current,
                             goal_month=min(state.goal_month, h))
        r2 = run_simulation(state, sc2, SimConfig(n_paths=2500, horizon_months=h))
        curve.append({"extra_monthly": extra, "goal_probability": r2.summary["goal_probability"]})

    return {
        "goal": {
            "target_amount": round(state.goal_target, 2),
            "current_amount": round(state.goal_current, 2),
            "months_remaining": int(state.goal_month),
            "progress_pct": round(float(state.goal_current / max(state.goal_target, 1) * 100), 1),
        },
        "probability": s["goal_probability"],
        "expected_goal_fund": s["expected_goal_fund"],
        "bands": res.bands["goal_fund"],
        "months": res.bands["months"],
        "required_contribution": targets,
        "sensitivity": curve,
        "assumptions": res.assumptions,
    }


def settings_horizon(state: UserState) -> int:
    return max(min(state.goal_month, 120), 12)
