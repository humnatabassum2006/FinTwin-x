"""
Tool layer — the *only* place where numbers are produced.

The agentic layer (LLM) may decide which tool to call, but it never computes a
financial figure itself. Every tool is a thin, validated wrapper around the
analytics / ML / simulation modules, which is what keeps the copilot grounded
and auditable.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from agents.context import DataContext, get_context
from common.logging_utils import get_logger
from common.utils import money

log = get_logger("tools")

DISCLAIMER = ("These are model estimates from simulated assumptions, not financial advice "
              "and not a guarantee of future outcomes.")


# --------------------------------------------------------------------------------------
# 1. snapshot / analysis
# --------------------------------------------------------------------------------------
def get_user_snapshot(user_id: int, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    row = c.row(user_id)
    panel = c.panel_for(user_id).tail(12)
    return {
        "user_id": int(user_id),
        "persona": str(row.get("persona")),
        "segment": str(row.get("segment_label", "")),
        "as_of": str(row.get("as_of"))[:10],
        "monthly_income": round(float(row["monthly_income"]), 2),
        "monthly_expense": round(float(row["monthly_expense"]), 2),
        "monthly_emi": round(float(row["monthly_emi"]), 2),
        "net_cash_flow": round(float(row["net_cash_flow"]), 2),
        "savings_rate": round(float(row["savings_rate"]), 4),
        "liquid_assets": round(float(row["liquid_assets"]), 2),
        "emergency_fund_months": round(float(row["emergency_fund_months"]), 2),
        "debt_balance": round(float(row["debt_balance"]), 2),
        "debt_service_ratio": round(float(row["debt_service_ratio"]), 4),
        "net_worth": round(float(row["net_worth"]), 2),
        "health_score": round(float(row["dna_financial_health"]), 1),
        "risk_score_rule": round(float(row["dna_financial_risk"]), 1),
        "goal": {
            "target": round(float(row["goal_target"]), 2),
            "current": round(float(row["goal_current"]), 2),
            "months_left": int(row["goal_months_left"]),
            "progress_pct": round(float(row["goal_progress"]) * 100, 1),
        },
        "history": {
            "months": [str(pd.Timestamp(m).date()) for m in panel["year_month"]],
            "income": panel["income"].round(0).tolist(),
            "expense": panel["expense"].round(0).tolist(),
            "net_cash_flow": panel["net_cash_flow"].round(0).tolist(),
            "cash_balance": panel["cash_balance"].round(0).tolist(),
        },
        "disclaimer": DISCLAIMER,
    }


def calculate_cashflow(user_id: int, months: int = 12, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    p = c.panel_for(user_id).tail(int(months))
    if p.empty:
        return {"error": "no history for this user"}
    inc, exp, emi = float(p["income"].mean()), float(p["expense"].mean()), float(p["emi"].mean())
    net = inc - exp - emi
    return {
        "months": int(len(p)),
        "avg_income": round(inc, 2),
        "avg_expense": round(exp, 2),
        "avg_emi": round(emi, 2),
        "avg_net_cash_flow": round(net, 2),
        "savings_rate": round(net / max(inc, 1), 4),
        "volatility": round(float(p["net_cash_flow"].std() / max(abs(net), 1)), 3),
        "negative_months": int((p["net_cash_flow"] < 0).sum()),
        "trend": {
            "income_slope": round(float(np.polyfit(range(len(p)), p["income"], 1)[0]), 2),
            "expense_slope": round(float(np.polyfit(range(len(p)), p["expense"], 1)[0]), 2),
        },
        "formatted": {
            "income": money(inc), "expense": money(exp), "emi": money(emi), "net": money(net),
        },
    }


# --------------------------------------------------------------------------------------
# 2. forecasting
# --------------------------------------------------------------------------------------
def forecast_expenses(user_id: int, horizons=(1, 3, 6, 12), ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    out = c.forecast_store.forecast_expenses(int(user_id))
    return {
        "horizons": {str(h): out[h] for h in out if h in horizons},
        "interpretation": {
            "1": "predicted spend next month",
            "3": "predicted average monthly spend over the next quarter",
            "6": "predicted average monthly spend over the next 6 months",
            "12": "predicted average monthly spend over the next year",
        },
        "interval": "90% conformal prediction interval from held-out residuals",
        "disclaimer": DISCLAIMER,
    }


def forecast_cashflow(user_id: int, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    fc = c.forecast_store.forecast_cashflow(int(user_id), c.panel)
    return {"horizons": {str(k): v for k, v in fc.items()}, "disclaimer": DISCLAIMER}


# --------------------------------------------------------------------------------------
# 3. risk + explainability
# --------------------------------------------------------------------------------------
def calculate_risk(user_id: int, explain: bool = True, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    row = c.row(user_id)
    model = c.risk_model
    if model is None:
        return {"error": "risk model not trained — run `make train`"}
    X = pd.DataFrame([row])[model.features].astype(float)
    score = float(model.predict_score(X)[0])
    from ml.risk.risk_model import risk_band
    out = {
        "user_id": int(user_id),
        "risk_score": round(score, 1),
        "band": risk_band(score),
        "probability_of_stress_6m": round(score / 100, 4),
        "rule_based_score": round(float(row["dna_financial_risk"]), 1),
        "sub_scores": {
            "liquidity_risk": round(100 - float(row["dna_liquidity"]), 1),
            "debt_risk": round(100 - float(row["dna_debt_resilience"]), 1),
            "income_risk": round(100 - float(row["dna_income_stability"]), 1),
            "spending_risk": round(100 - float(row["dna_spending_stability"]), 1),
            "savings_risk": round(100 - float(row["dna_savings_discipline"]), 1),
            "emergency_risk": round(100 - float(row["dna_emergency_resilience"]), 1),
            "goal_risk": round(100 - float(row["dna_goal_discipline"]), 1),
        },
        "disclaimer": DISCLAIMER,
    }
    if explain:
        from ml.explainability.shap_explain import explain as shap_explain
        try:
            out["explanation"] = shap_explain(model, X, c.background)
        except Exception as exc:
            log.warning("explanation failed: %s", exc)
            out["explanation"] = None
    return out


def explain_risk(user_id: int, ctx: DataContext | None = None) -> dict:
    r = calculate_risk(user_id, explain=True, ctx=ctx)
    return {"risk_score": r["risk_score"], "band": r["band"],
            "contributions": (r.get("explanation") or {}).get("contributions", []),
            "method": (r.get("explanation") or {}).get("method"),
            "top_driver": (r.get("explanation") or {}).get("top_driver")}


def counterfactual_options(user_id: int, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    model = c.risk_model
    row = c.row(user_id)
    from ml.explainability.counterfactual import counterfactuals, plan_to_target
    cf = counterfactuals(model, row)
    plan = plan_to_target(model, row, target_score=40.0)
    return {"current_score": cf["current_score"], "options": cf["options"],
            "best_single_action": cf["best"], "plan_to_reach_40": plan, "disclaimer": DISCLAIMER}


# --------------------------------------------------------------------------------------
# 4. simulation
# --------------------------------------------------------------------------------------
def run_monte_carlo(user_id: int, scenario: dict | None = None, n_paths: int = 5000,
                    horizon_months: int = 36, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    row = c.row(user_id)
    from simulation.scenario_engine import ScenarioSpec, run_scenario
    from simulation.monte_carlo import SimConfig
    spec = ScenarioSpec(**{**(scenario or {}), "n_paths": n_paths, "horizon_months": horizon_months})
    cfg = SimConfig(n_paths=n_paths, horizon_months=horizon_months)
    from simulation.scenario_engine import run_scenario as _run
    from simulation.monte_carlo import ScenarioParams
    state = _user_state(row)
    res = _run(row, spec, cfg)
    return res


def _user_state(row):
    from simulation.monte_carlo import UserState
    return UserState.from_features(row)


def compare_scenarios(user_id: int, scenarios: list[dict] | None = None,
                      n_paths: int = 4000, horizon_months: int = 36,
                      ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    row = c.row(user_id)
    from simulation.monte_carlo import SimConfig
    from simulation.scenario_engine import ScenarioSpec, compare_scenarios as _cmp
    specs = [ScenarioSpec(name="baseline", label="Current path",
                          n_paths=n_paths, horizon_months=horizon_months)]
    for s in (scenarios or []):
        specs.append(ScenarioSpec(**{**s, "n_paths": n_paths, "horizon_months": horizon_months}))
    return _cmp(row, specs, SimConfig(n_paths=n_paths, horizon_months=horizon_months))


def optimise_decision(user_id: int, scenario: dict | None = None, target_metric: str = "goal_probability",
                      target_value: float = 0.75, n_paths: int = 2500,
                      horizon_months: int = 36, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    row = c.row(user_id)
    from simulation.monte_carlo import SimConfig
    from simulation.scenario_engine import ScenarioSpec, optimise_decision as _opt
    spec = ScenarioSpec(**{**(scenario or {}), "n_paths": n_paths, "horizon_months": horizon_months})
    return _opt(row, spec, target_metric=target_metric, target_value=target_value,
                cfg=SimConfig(n_paths=n_paths, horizon_months=horizon_months))


def stress_test(user_id: int, horizon_months: int = 36, n_paths: int = 3000,
                ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    from simulation.stress_testing import stress_matrix
    return stress_matrix(c.row(user_id), horizon=horizon_months, n_paths=n_paths)


def goal_outlook(user_id: int, n_paths: int = 4000, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    from simulation.goal import goal_outlook as _goal
    return _goal(c.row(user_id), n_paths=n_paths)


# --------------------------------------------------------------------------------------
# 5. anomalies & spending intelligence
# --------------------------------------------------------------------------------------
def detect_anomalies(user_id: int, top_k: int = 8, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    tx = c.tx_for(user_id)
    model = c.anomaly_model
    from ml.anomaly_detection.anomaly import category_month_alerts, user_anomalies
    items = user_anomalies(model, tx, int(user_id), top_k=top_k) if model is not None else []
    return {
        "transactions_scanned": int(len(tx)),
        "top_anomalies": items,
        "category_alerts": category_month_alerts(tx, int(user_id)),
        "method": "IsolationForest + robust MAD z-score + burst detector (fused rank score)",
    }


def spending_intelligence(user_id: int, ctx: DataContext | None = None) -> dict:
    c = ctx or get_context()
    tx = c.tx_for(user_id).copy()
    p = c.panel_for(user_id).tail(12)
    if tx.empty:
        return {"error": "no transactions"}
    tx["ym"] = tx["timestamp"].to_numpy().astype("datetime64[M]")
    by_cat = tx.groupby("category", observed=True)["amount"].agg(["sum", "mean", "count"])
    total = float(by_cat["sum"].sum())
    merchants = tx.groupby("merchant", observed=False)["amount"].sum().sort_values(ascending=False)
    hhi = float(((merchants / merchants.sum()) ** 2).sum())
    tx["dow"] = tx["timestamp"].dt.dayofweek
    weekend = float(tx.loc[tx["dow"].isin([5, 6]), "amount"].sum() / max(total, 1))
    tx["hour"] = tx["timestamp"].dt.hour
    latenight = float(tx.loc[tx["hour"] >= 23, "amount"].sum() / max(total, 1))

    months = [str(pd.Timestamp(m).date()) for m in p["year_month"]]
    cat_series = {c_: p[f"exp_{c_}"].round(0).tolist() for c_ in
                  ["Housing", "Food", "Transport", "Utilities", "Shopping", "Entertainment",
                   "Education", "Healthcare", "Subscriptions", "Other"] if f"exp_{c_}" in p.columns}

    return {
        "total_spend_12m": round(total, 2),
        "categories": [
            {"category": str(i), "amount": round(float(r["sum"]), 2),
             "share": round(float(r["sum"]) / max(total, 1), 4),
             "avg_tx": round(float(r["mean"]), 2), "count": int(r["count"])}
            for i, r in by_cat.sort_values("sum", ascending=False).iterrows()
        ],
        "top_merchants": [{"merchant": str(k), "amount": round(float(v), 2)}
                          for k, v in merchants.head(8).items()],
        "merchant_concentration_hhi": round(hhi, 4),
        "weekend_spend_share": round(weekend, 4),
        "latenight_spend_share": round(latenight, 4),
        "monthly_by_category": {"months": months, "series": cat_series},
        "subscription_monthly": round(float(p["exp_Subscriptions"].mean()) if "exp_Subscriptions" in p else 0.0, 2),
    }


# --------------------------------------------------------------------------------------
# 6. knowledge (RAG)
# --------------------------------------------------------------------------------------
def knowledge_search(query: str, k: int = 4, ctx: DataContext | None = None) -> dict:
    from rag.retrieval.retriever import get_retriever
    r = get_retriever()
    hits = r.search(query, k=k)
    return {"query": query, "hits": hits, "backend": r.backend}


# --------------------------------------------------------------------------------------
# registry (used by the LLM tool-calling prompt and by /tools)
# --------------------------------------------------------------------------------------
TOOLS = {
    "get_user_snapshot": {
        "fn": get_user_snapshot,
        "description": "Current financial snapshot: income, expenses, savings rate, liquidity, goals.",
        "parameters": {"user_id": "int"},
    },
    "calculate_cashflow": {
        "fn": calculate_cashflow,
        "description": "Historical cash-flow analysis for the last N months.",
        "parameters": {"user_id": "int", "months": "int"},
    },
    "forecast_expenses": {
        "fn": forecast_expenses,
        "description": "Predicted expenses at 1, 3, 6 and 12 months with 90% intervals.",
        "parameters": {"user_id": "int"},
    },
    "forecast_cashflow": {
        "fn": forecast_cashflow,
        "description": "Forecast income, expense, EMI and net cash flow at 1/3/6/12 months.",
        "parameters": {"user_id": "int"},
    },
    "calculate_risk": {
        "fn": calculate_risk,
        "description": "FinTwin Risk Score 0-100 = modelled probability of financial stress in 6 months.",
        "parameters": {"user_id": "int"},
    },
    "explain_risk": {
        "fn": explain_risk,
        "description": "SHAP decomposition of the risk score into per-driver risk points.",
        "parameters": {"user_id": "int"},
    },
    "counterfactual_options": {
        "fn": counterfactual_options,
        "description": "What the risk score becomes under each concrete action (counterfactuals).",
        "parameters": {"user_id": "int"},
    },
    "run_monte_carlo": {
        "fn": run_monte_carlo,
        "description": "Run thousands of simulated futures for a scenario (what-if).",
        "parameters": {"user_id": "int", "scenario": "object", "n_paths": "int", "horizon_months": "int"},
    },
    "compare_scenarios": {
        "fn": compare_scenarios,
        "description": "Compare several what-if scenarios side by side.",
        "parameters": {"user_id": "int", "scenarios": "array"},
    },
    "optimise_decision": {
        "fn": optimise_decision,
        "description": "Search decision alternatives (delay, smaller ticket, income boost, cut spend) and rank them.",
        "parameters": {"user_id": "int", "scenario": "object", "target_metric": "string", "target_value": "number"},
    },
    "stress_test": {
        "fn": stress_test,
        "description": "Run a battery of adverse scenarios (job loss, inflation, medical, EMI reset).",
        "parameters": {"user_id": "int"},
    },
    "goal_outlook": {
        "fn": goal_outlook,
        "description": "Probability of reaching the primary goal and the contribution required.",
        "parameters": {"user_id": "int"},
    },
    "detect_anomalies": {
        "fn": detect_anomalies,
        "description": "Unusual transactions and category-level spending alerts.",
        "parameters": {"user_id": "int", "top_k": "int"},
    },
    "spending_intelligence": {
        "fn": spending_intelligence,
        "description": "Category breakdown, merchant concentration, behavioural patterns.",
        "parameters": {"user_id": "int"},
    },
    "knowledge_search": {
        "fn": knowledge_search,
        "description": "Retrieve financial-literacy / planning knowledge from the RAG knowledge base.",
        "parameters": {"query": "string", "k": "int"},
    },
}


def call_tool(name: str, **kwargs) -> Any:
    if name not in TOOLS:
        raise KeyError(f"unknown tool '{name}'")
    return TOOLS[name]["fn"](**kwargs)


def tool_manifest() -> list[dict]:
    return [{"name": n, "description": t["description"], "parameters": t["parameters"]}
            for n, t in TOOLS.items()]
