"""
Monte-Carlo engine — the probabilistic core of FinTwin-X.

Instead of one deterministic projection we simulate thousands of coherent
futures. Every path draws its own inflation regime, income growth, investment
returns and shock sequence, then evolves the full household balance sheet
month by month.

Determinism: the same seed always produces the same paths, which is what makes
the engine testable and the results reproducible (see tests/test_simulation.py).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import settings
from common.logging_utils import get_logger
from common.utils import cvar, emi

log = get_logger("simulation")


# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
@dataclass
class SimConfig:
    n_paths: int = settings.N_SIMULATIONS
    horizon_months: int = 60
    seed: int = settings.RANDOM_SEED

    # macro / market assumptions (annualised)
    inflation_mu: float = 0.075
    inflation_sigma: float = 0.020
    income_growth_mu: float = 0.060
    income_growth_sigma: float = 0.025
    income_noise: float = 0.070
    expense_noise: float = 0.110
    invest_return_mu: float = 0.110
    invest_return_sigma: float = 0.180
    invest_return_risk_free: float = 0.145      # fixed-deposit / sukuk alternative

    # behavioural
    shock_prob: float = 0.020                   # monthly probability of an unexpected expense
    shock_size_mu: float = 0.9                  # in months of expenses
    shock_size_sigma: float = 0.6
    liquid_share: float = 0.75                  # share of investments that are liquid

    # stress definition
    stress_liquidity_months: float = 0.0        # stressed when liquid assets < 0
    goal_month: int | None = None               # month in which the goal is evaluated


@dataclass
class ScenarioParams:
    """Everything a what-if can change. Defaults = baseline (no change)."""
    name: str = "baseline"
    label: str = "Current path"
    horizon_months: int | None = None

    income_change_pct: float = 0.0              # permanent income delta
    income_shock_pct: float = 0.0               # temporary income delta (e.g. -1.0 = job loss)
    shock_start_month: int = 1
    shock_duration_months: int = 0
    recovery_factor: float = 1.0                # income multiplier after the shock ends
    extra_monthly_income: float = 0.0

    expense_change_pct: float = 0.0
    inflation_rate: float | None = None         # overrides the stochastic inflation regime
    lump_sum_expense: float = 0.0
    lump_sum_month: int = 1

    monthly_investment: float | None = None     # absolute monthly contribution
    invest_return_mu: float | None = None

    new_debt_principal: float = 0.0
    new_debt_rate: float = 0.20
    new_debt_term_months: int = 60
    new_debt_start_month: int = 1
    debt_payment_multiplier: float = 1.0

    goal_target: float | None = None
    goal_current: float = 0.0
    goal_month: int | None = None
    monthly_goal_contribution: float | None = None

    def describe(self) -> list[str]:
        bits = []
        if self.income_change_pct:
            bits.append(f"income {self.income_change_pct:+.0%}")
        if self.income_shock_pct:
            bits.append(f"income shock {self.income_shock_pct:+.0%} for {self.shock_duration_months} months")
        if self.extra_monthly_income:
            bits.append(f"+Rs {self.extra_monthly_income:,.0f}/month")
        if self.expense_change_pct:
            bits.append(f"expenses {self.expense_change_pct:+.0%}")
        if self.inflation_rate is not None:
            bits.append(f"inflation {self.inflation_rate:.1%}")
        if self.lump_sum_expense:
            bits.append(f"one-off Rs {self.lump_sum_expense:,.0f}")
        if self.new_debt_principal:
            bits.append(f"new debt Rs {self.new_debt_principal:,.0f} @ {self.new_debt_rate:.0%}")
        if self.monthly_investment:
            bits.append(f"invest Rs {self.monthly_investment:,.0f}/month")
        if self.debt_payment_multiplier != 1.0:
            bits.append(f"EMI x{self.debt_payment_multiplier}")
        return bits or ["no change"]


# --------------------------------------------------------------------------------------
# user state
# --------------------------------------------------------------------------------------
@dataclass
class UserState:
    monthly_income: float
    monthly_expense: float
    monthly_emi: float = 0.0
    cash: float = 0.0
    investments: float = 0.0
    debt_balance: float = 0.0
    debt_rate: float = 0.20
    debt_payment: float = 0.0
    goal_target: float = 0.0
    goal_current: float = 0.0
    goal_month: int = 36
    monthly_investment: float = 0.0
    liquid_share: float = 0.75

    @classmethod
    def from_features(cls, row, horizon: int = 60) -> "UserState":
        def g(k, d=0.0):
            v = row.get(k, d) if hasattr(row, "get") else getattr(row, k, d)
            try:
                return float(v)
            except (TypeError, ValueError):
                return d
        return cls(
            monthly_income=max(g("monthly_income"), 1.0),
            monthly_expense=max(g("monthly_expense"), 1.0),
            monthly_emi=max(g("monthly_emi"), 0.0),
            cash=g("cash_balance"),
            investments=g("invest_value"),
            debt_balance=g("debt_balance"),
            debt_rate=g("avg_interest_rate", 0.20) or 0.20,
            debt_payment=max(g("monthly_emi"), 0.0),
            goal_target=g("goal_target"),
            goal_current=g("goal_current"),
            goal_month=max(int(g("goal_months_left", min(horizon, 36))), 1),
            monthly_investment=g("monthly_surplus") * 0.5,
            liquid_share=0.75,
        )


# --------------------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------------------
@dataclass
class SimulationResult:
    name: str
    label: str
    summary: dict
    bands: dict
    histogram: dict
    assumptions: dict
    paths_shape: tuple


def run_simulation(state: UserState, scenario: ScenarioParams | None = None,
                   cfg: SimConfig | None = None) -> SimulationResult:
    cfg = cfg or SimConfig()
    sc = scenario or ScenarioParams()
    horizon = int(sc.horizon_months or cfg.horizon_months)
    n = int(cfg.n_paths)
    rng = np.random.default_rng(cfg.seed)

    # ---- per-path regime draws ---------------------------------------------------
    infl_annual = rng.normal(cfg.inflation_mu, cfg.inflation_sigma, size=n) if sc.inflation_rate is None \
        else np.full(n, sc.inflation_rate)
    infl_m = (1 + np.clip(infl_annual, -0.02, 0.5)) ** (1 / 12) - 1

    growth_annual = rng.normal(cfg.income_growth_mu, cfg.income_growth_sigma, size=n)
    growth_m = (1 + np.clip(growth_annual, -0.2, 0.5)) ** (1 / 12) - 1

    ret_mu = sc.invest_return_mu if sc.invest_return_mu is not None else cfg.invest_return_mu
    ret_annual = rng.normal(ret_mu, cfg.invest_return_sigma * 0.6, size=n)
    ret_m = (1 + np.clip(ret_annual, -0.6, 1.0)) ** (1 / 12) - 1

    # ---- state arrays --------------------------------------------------------------
    cash = np.full(n, float(state.cash))
    inv = np.full(n, float(state.investments))
    debt = np.full(n, float(state.debt_balance))
    goal_fund = np.full(n, float(state.goal_current))
    durable = np.zeros(n)                       # asset bought with new debt (depreciating)

    base_income = float(state.monthly_income)
    base_expense = float(state.monthly_expense) * (1 + sc.expense_change_pct)
    emi_base = float(state.debt_payment) * sc.debt_payment_multiplier
    debt_rate_m = float(state.debt_rate) / 12
    new_debt = np.zeros(n)
    new_emi = np.zeros(n)

    monthly_invest = float(sc.monthly_investment) if sc.monthly_investment is not None \
        else max(float(state.monthly_investment), 0.0)

    goal_target = float(sc.goal_target if sc.goal_target is not None else state.goal_target)
    goal_month = max(1, min(int(sc.goal_month if sc.goal_month is not None else state.goal_month), horizon))
    required_goal = max(goal_target - float(state.goal_current), 0.0) / max(goal_month, 1) if goal_target > 0 else 0.0

    # overdraft / credit-card ceiling: nobody can spend infinitely
    credit_floor = -2.0 * max(base_income, 1.0)

    # ---- path storage ---------------------------------------------------------------
    net_worth_path = np.zeros((horizon, n), dtype=np.float32)
    liquid_path = np.zeros((horizon, n), dtype=np.float32)
    cashflow_path = np.zeros((horizon, n), dtype=np.float32)
    goal_path = np.zeros((horizon, n), dtype=np.float32)
    stressed = np.zeros(n, dtype=bool)
    ruin_month = np.full(n, -1, dtype=np.int16)

    cum_infl = np.ones(n)
    cum_growth = np.ones(n)

    for t in range(horizon):
        cum_infl *= (1 + infl_m)
        cum_growth *= (1 + growth_m)

        # --- income -------------------------------------------------------------------
        noise_inc = rng.normal(0, cfg.income_noise, size=n)
        inc = np.maximum(base_income * cum_growth * (1 + sc.income_change_pct) * (1 + noise_inc), 0.0)
        inc = inc + sc.extra_monthly_income
        if sc.income_shock_pct and sc.shock_duration_months > 0:
            m0 = sc.shock_start_month - 1
            if m0 <= t < m0 + sc.shock_duration_months:
                inc = inc * (1 + sc.income_shock_pct)
            elif t >= m0 + sc.shock_duration_months:
                inc = inc * sc.recovery_factor

        # --- expenses (with behavioural curtailment when buffers run thin) ------------
        noise_exp = rng.normal(0, cfg.expense_noise, size=n)
        exp = np.maximum(base_expense * cum_infl * (1 + noise_exp), 0.0)
        cover = (cash + inv * cfg.liquid_share) / np.maximum(exp, 1.0)
        curtail = np.clip(0.60 + 0.40 * cover, 0.60, 1.0)
        exp = exp * np.where(cover < 1.0, curtail, 1.0)

        # shocks (medical, appliance, family events)
        shock_hit = rng.random(n) < cfg.shock_prob
        exp = exp + np.where(shock_hit, base_expense * np.abs(rng.lognormal(
            cfg.shock_size_mu - 0.5 * cfg.shock_size_sigma ** 2, cfg.shock_size_sigma, size=n)), 0.0)

        # --- financed purchase ---------------------------------------------------------
        if sc.new_debt_principal > 0 and t == max(sc.new_debt_start_month - 1, 0):
            new_debt[:] = sc.new_debt_principal
            new_emi[:] = emi(sc.new_debt_principal, sc.new_debt_rate, sc.new_debt_term_months)
            durable[:] = sc.new_debt_principal + sc.lump_sum_expense
        durable *= (1 - 0.12) ** (1 / 12)          # depreciation of the financed asset

        # --- one-off expense (paid from cash, then investments, then the goal fund) ----
        if sc.lump_sum_expense and t == max(sc.lump_sum_month - 1, 0):
            need = np.full(n, float(sc.lump_sum_expense))
            pay = np.minimum(np.maximum(cash, 0.0), need)
            cash -= pay
            need -= pay
            sell = np.minimum(inv * cfg.liquid_share, need)
            inv -= sell
            need -= sell
            take = np.minimum(goal_fund, need)
            goal_fund -= take
            need -= take
            cash -= need                              # remainder = distress borrowing

        # --- debt service ---------------------------------------------------------------
        emi_total = emi_base + new_emi
        new_debt = np.maximum(new_debt * (1 + sc.new_debt_rate / 12) - new_emi, 0.0)
        debt = np.maximum(debt * (1 + debt_rate_m) - emi_base, 0.0)

        # --- cash flow & allocation --------------------------------------------------------
        net = inc - exp - emi_total
        cash = np.maximum(cash + net, credit_floor)

        surplus = np.maximum(net, 0.0)
        # the goal is funded from this month's surplus, plus a gradual deployment of
        # cash held ABOVE a two-month buffer (people do save towards goals from stock)
        excess_cash = np.maximum(cash - 2.0 * np.maximum(exp, 1.0), 0.0)
        if sc.monthly_goal_contribution is not None:
            # an explicit contribution plan overrides the automatic rule
            goal_contrib = np.minimum(float(sc.monthly_goal_contribution),
                                      np.maximum(cash, 0.0) + surplus)
        else:
            goal_contrib = np.where(goal_target > 0,
                                    np.minimum(required_goal, surplus + 0.08 * excess_cash), 0.0)
        other_invest = np.where(sc.monthly_investment is not None,
                                np.minimum(monthly_invest, np.maximum(cash - goal_contrib, 0.0)),
                                np.minimum(np.maximum(surplus - goal_contrib, 0.0) * 0.5,
                                           np.maximum(cash - goal_contrib, 0.0)))
        cash -= (goal_contrib + other_invest)
        inv = inv * (1 + ret_m) + other_invest
        goal_fund = goal_fund * (1 + ret_m) + goal_contrib

        liquid = cash + inv * cfg.liquid_share
        net_worth = cash + inv + durable - debt - new_debt

        broke = (liquid < 0) | (cash <= credit_floor + 1e-6)
        newly = broke & ~stressed
        if newly.any():
            ruin_month[newly] = t
        stressed |= broke

        net_worth_path[t] = net_worth
        liquid_path[t] = liquid
        cashflow_path[t] = net
        goal_path[t] = goal_fund

    # ---- summaries ---------------------------------------------------------------
    final_nw = net_worth_path[-1]
    final_liquid = liquid_path[-1]
    goal_at_month = goal_path[goal_month - 1] if goal_month <= horizon else goal_path[-1]
    goal_hit = goal_at_month >= goal_target if goal_target > 0 else np.zeros(n, dtype=bool)

    def q(a, p):
        return float(np.percentile(a, p))

    summary = {
        "goal_probability": round(float(goal_hit.mean()), 4) if goal_target > 0 else None,
        "goal_target": round(goal_target, 2),
        "goal_month": goal_month,
        "expected_goal_fund": round(float(goal_at_month.mean()), 2),
        "stress_probability": round(float(stressed.mean()), 4),
        "expected_final_net_worth": round(float(final_nw.mean()), 2),
        "median_final_net_worth": round(q(final_nw, 50), 2),
        "p5_final_net_worth": round(q(final_nw, 5), 2),
        "p95_final_net_worth": round(q(final_nw, 95), 2),
        "cvar_5_final_net_worth": round(cvar(final_nw, 0.05), 2),
        "expected_final_liquid": round(float(final_liquid.mean()), 2),
        "p5_final_liquid": round(q(final_liquid, 5), 2),
        "median_months_of_cover_at_end": round(
            float(np.median(final_liquid / max(base_expense, 1.0))), 2),
        "worst_case_month": int(np.median(ruin_month[ruin_month >= 0])) if (ruin_month >= 0).any() else None,
        "probability_negative_cash_flow_any_month": round(
            float((cashflow_path < 0).any(axis=0).mean()), 4),
        "expected_final_debt": round(float((debt + new_debt).mean()), 2),
    }

    bands = {
        "months": list(range(1, horizon + 1)),
        "net_worth": {f"p{p}": np.percentile(net_worth_path, p, axis=1).round(1).tolist() for p in (5, 25, 50, 75, 95)},
        "liquid_assets": {f"p{p}": np.percentile(liquid_path, p, axis=1).round(1).tolist() for p in (5, 25, 50, 75, 95)},
        "net_cash_flow": {f"p{p}": np.percentile(cashflow_path, p, axis=1).round(1).tolist() for p in (5, 50, 95)},
        "goal_fund": {f"p{p}": np.percentile(goal_path, p, axis=1).round(1).tolist() for p in (5, 50, 95)},
        "goal_target": goal_target,
    }

    if np.ptp(final_nw) < 1e-9:                       # degenerate path (zero variance)
        final_nw = final_nw + rng.normal(0, max(abs(final_nw.mean()) * 0.01, 1.0), size=n)
    hist_counts, hist_edges = np.histogram(final_nw, bins=40)
    histogram = {"counts": hist_counts.tolist(), "edges": hist_edges.round(0).tolist()}

    assumptions = {
        "n_paths": n,
        "horizon_months": horizon,
        "seed": cfg.seed,
        "inflation_mu": sc.inflation_rate if sc.inflation_rate is not None else cfg.inflation_mu,
        "income_growth_mu": cfg.income_growth_mu,
        "invest_return_mu": ret_mu,
        "shock_prob": cfg.shock_prob,
        "changes": sc.describe(),
    }
    log.debug("simulation %s done: %s paths × %s months", sc.name, f"{n:,}", horizon)
    return SimulationResult(name=sc.name, label=sc.label, summary=summary, bands=bands,
                            histogram=histogram, assumptions=assumptions, paths_shape=(horizon, n))


def required_monthly_contribution(state: UserState, target_probability: float = 0.75,
                                  cfg: SimConfig | None = None, lo: float = 0.0,
                                  hi: float | None = None, tol: float = 500.0) -> dict:
    """Bisection search for the monthly contribution that hits the target probability."""
    cfg = cfg or SimConfig(n_paths=max(2_000, cfg.n_paths if cfg else 2_000))
    hi = hi or max(state.monthly_income * 0.6, 10_000)

    def prob(c: float) -> float:
        sc = ScenarioParams(name="search", monthly_investment=c, monthly_goal_contribution=c,
                            goal_target=state.goal_target, goal_current=state.goal_current,
                            goal_month=state.goal_month, horizon_months=max(state.goal_month, 12))
        res = run_simulation(state, sc, cfg)
        return res.summary["goal_probability"] or 0.0

    p_lo, p_hi = prob(lo), prob(hi)
    if p_hi < target_probability:
        return {"required_monthly_contribution": None, "achieved_probability": round(p_hi, 3),
                "feasible": False, "note": "Target not reachable within 60% of income."}
    for _ in range(14):
        mid = (lo + hi) / 2
        p = prob(mid)
        if p < target_probability:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return {"required_monthly_contribution": round(hi, -2), "achieved_probability": round(prob(hi), 3),
            "feasible": True, "note": "Contribution redirected to the goal fund each month."}
