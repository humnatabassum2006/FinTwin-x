"""Simulation engine: determinism, monotonicity, boundary conditions."""
from __future__ import annotations

import numpy as np
import pytest

from simulation.monte_carlo import (ScenarioParams, SimConfig, UserState,
                                    run_simulation)
from simulation.scenario_engine import ScenarioSpec, compare_scenarios, optimise_decision
from simulation.stress_testing import stress_matrix


@pytest.fixture(scope="module")
def state():
    return UserState(monthly_income=200_000, monthly_expense=130_000, monthly_emi=20_000,
                     cash=600_000, investments=500_000, debt_balance=800_000,
                     debt_rate=0.18, debt_payment=20_000, goal_target=3_000_000,
                     goal_current=400_000, goal_month=36)


def test_simulation_is_reproducible(state):
    a = run_simulation(state, ScenarioParams(name="a"), SimConfig(n_paths=800, horizon_months=24, seed=1))
    b = run_simulation(state, ScenarioParams(name="b"), SimConfig(n_paths=800, horizon_months=24, seed=1))
    assert a.summary == b.summary
    assert np.allclose(a.bands["net_worth"]["p50"], b.bands["net_worth"]["p50"])


def test_different_seed_changes_paths(state):
    a = run_simulation(state, ScenarioParams(), SimConfig(n_paths=800, horizon_months=24, seed=1))
    b = run_simulation(state, ScenarioParams(), SimConfig(n_paths=800, horizon_months=24, seed=2))
    assert a.summary["expected_final_net_worth"] != b.summary["expected_final_net_worth"]


def test_more_debt_raises_stress(state):
    base = run_simulation(state, ScenarioParams(), SimConfig(n_paths=1500, horizon_months=36)).summary
    car = run_simulation(state, ScenarioParams(new_debt_principal=3_000_000, new_debt_rate=0.20,
                                               new_debt_term_months=60, lump_sum_expense=1_000_000),
                         SimConfig(n_paths=1500, horizon_months=36)).summary
    assert car["stress_probability"] >= base["stress_probability"]


def test_job_loss_raises_stress(state):
    base = run_simulation(state, ScenarioParams(), SimConfig(n_paths=1500, horizon_months=36)).summary
    job = run_simulation(state, ScenarioParams(income_shock_pct=-1.0, shock_duration_months=6,
                                               recovery_factor=0.9),
                         SimConfig(n_paths=1500, horizon_months=36)).summary
    assert job["stress_probability"] > base["stress_probability"]


def test_boundary_zero_income_always_stresses(state):
    """Extreme scenario: no income for the whole horizon ⇒ near-certain stress."""
    res = run_simulation(state, ScenarioParams(income_shock_pct=-1.0, shock_duration_months=120),
                         SimConfig(n_paths=500, horizon_months=36))
    assert res.summary["stress_probability"] > 0.9


def test_zero_horizon_and_single_path(state):
    res = run_simulation(state, ScenarioParams(), SimConfig(n_paths=1, horizon_months=1))
    assert len(res.bands["months"]) == 1
    assert np.isfinite(res.summary["expected_final_net_worth"])


def test_goal_probability_is_a_probability(state):
    res = run_simulation(state, ScenarioParams(), SimConfig(n_paths=1500, horizon_months=36))
    p = res.summary["goal_probability"]
    assert p is not None and 0.0 <= p <= 1.0


def test_percentiles_are_ordered(state):
    res = run_simulation(state, ScenarioParams(), SimConfig(n_paths=1500, horizon_months=36))
    nw = res.bands["net_worth"]
    assert all(nw["p5"][i] <= nw["p50"][i] <= nw["p95"][i] for i in range(len(nw["p5"])))


def test_scenario_comparison_shape():
    import pandas as pd
    row = pd.Series({"monthly_income": 200_000, "monthly_expense": 130_000, "monthly_emi": 20_000,
                     "cash_balance": 600_000, "liquid_assets": 900_000, "invest_value": 500_000,
                     "debt_balance": 800_000, "avg_interest_rate": 0.18, "goal_target": 3_000_000,
                     "goal_current": 400_000, "goal_months_left": 36, "monthly_surplus": 50_000,
                     "user_id": 1})
    base = ScenarioSpec(name="baseline", label="Current path", n_paths=800, horizon_months=24)
    car = ScenarioSpec(name="car", label="Car", new_debt_principal=3_000_000,
                       lump_sum_expense=1_000_000, n_paths=800, horizon_months=24)
    out = compare_scenarios(row, [base, car], None)
    assert len(out["scenarios"]) == 2
    assert any(r["metric"] == "goal_probability" for r in out["table"])
    assert "curves" in out


def test_optimiser_returns_ranked_options(state):
    import pandas as pd
    row = pd.Series({"monthly_income": 200_000, "monthly_expense": 130_000, "monthly_emi": 20_000,
                     "cash_balance": 600_000, "liquid_assets": 900_000, "invest_value": 500_000,
                     "debt_balance": 800_000, "avg_interest_rate": 0.18, "goal_target": 3_000_000,
                     "goal_current": 400_000, "goal_months_left": 36, "monthly_surplus": 50_000,
                     "user_id": 1})
    spec = ScenarioSpec(name="car", label="Car", new_debt_principal=3_000_000,
                        lump_sum_expense=1_000_000, n_paths=800, horizon_months=36)
    from simulation.monte_carlo import SimConfig
    out = optimise_decision(row, spec, cfg=SimConfig(n_paths=800, horizon_months=36))
    assert len(out["options"]) >= 3
    probs = [o["summary"].get("goal_probability") or -1 for o in out["options"]]
    assert probs == sorted(probs, reverse=True)
    assert set(out) >= {"options", "baseline", "recommended"}


def test_stress_matrix_runs(state):
    import pandas as pd
    row = pd.Series({"monthly_income": 200_000, "monthly_expense": 130_000, "monthly_emi": 20_000,
                     "cash_balance": 600_000, "liquid_assets": 900_000, "invest_value": 500_000,
                     "debt_balance": 800_000, "avg_interest_rate": 0.18, "goal_target": 3_000_000,
                     "goal_current": 400_000, "goal_months_left": 36, "monthly_surplus": 50_000,
                     "user_id": 1})
    out = stress_matrix(row, horizon=24, n_paths=400)
    assert len(out["tests"]) >= 8
    assert all(0 <= t["stress_probability"] <= 1 for t in out["tests"])
