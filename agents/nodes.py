"""
Specialised agent nodes.

  Financial Analyst  — snapshot, cash flow, spending intelligence
  Risk Analyst       — risk score, SHAP attribution, counterfactuals
  Scenario Analyst   — parses "what if" language into a simulation spec
  Simulation Agent   — runs the Monte-Carlo engine (never the LLM)
  Research Agent     — RAG retrieval over the financial knowledge base
  Decision Agent     — searches alternatives and ranks them
  Explanation Agent  — turns tool output into honest, hedged language
"""
from __future__ import annotations

import re

from agents import tools
from agents.context import get_context
from common.logging_utils import get_logger

log = get_logger("agents")

INTENT_RULES = [
    ("optimise", r"\b(safest|best way|optimi[sz]e|alternative|instead of|should i|recommend)\b"),
    ("anomaly", r"\b(unusual|anomal|fraud|suspicious|weird|odd|outlier)\b"),
    ("goal", r"\b(goal|target|save for|retire|retirement|down payment|umrah|hajj|education fund)\b"),
    ("risk", r"\b(risk|fragile|danger|safe|resilien|how bad|vulnerab)\b"),
    ("forecast", r"\b(forecast|predict|next month|next year|next 12|future|will my|expect)\b"),
    ("scenario", r"\b(what if|can i afford|afford|buy|purchase|loan|car|house|home|job loss|lose my|"
                 r"salary cut|inflation|medical|emergency|debt)\b"),
    ("conceptual", r"\b(what is|explain|how does|why should|difference between|define|meaning of|"
                   r"how do i|methodolog)\b"),
    ("snapshot", r"\b(overview|summary|how am i doing|health|cash flow|snapshot|status|spending)\b"),
]

PLANS = {
    "conceptual": ["research"],
    "snapshot": ["analyst"],
    "forecast": ["analyst", "forecast"],
    "risk": ["analyst", "risk"],
    "goal": ["analyst", "goal"],
    "anomaly": ["analyst", "anomaly"],
    "scenario": ["analyst", "risk", "scenario", "decision"],
    "optimise": ["analyst", "risk", "scenario", "decision"],
}

AMOUNT_PATTERNS = [
    (r"(\d+(?:\.\d+)?)\s*crore", 1e7),
    (r"(\d+(?:\.\d+)?)\s*(?:million|m\b)", 1e6),
    (r"(\d+(?:\.\d+)?)\s*lakh", 1e5),
    (r"(\d+(?:\.\d+)?)\s*k\b", 1e3),
    (r"rs\.?\s*(\d{1,3}(?:,\d{2,3})+)", 1.0),
    (r"\b(\d{5,9})\b", 1.0),
]


def parse_amount(text: str) -> float | None:
    t = text.lower()
    for pat, mult in AMOUNT_PATTERNS:
        m = re.search(pat, t)
        if m:
            raw = m.group(1).replace(",", "")
            try:
                return float(raw) * mult
            except ValueError:
                continue
    return None


def parse_percentage(text: str) -> float | None:
    m = re.search(r"(\d{1,3})\s*%", text)
    return float(m.group(1)) / 100 if m else None


def parse_months(text: str) -> int | None:
    m = re.search(r"(\d{1,2})\s*(?:month|months|mah)", text.lower())
    return int(m.group(1)) if m else None


HYPOTHETICAL = r"\b(what happens if|what if|suppose|imagine|let'?s say|if i |if my |assume)\b"


def detect_intent(question: str) -> str:
    q = question.lower()
    # explicit hypotheticals always go to the simulation agents
    if re.search(HYPOTHETICAL, q):
        return "optimise" if re.search(r"\b(safest|best way|should i|optimi[sz]e|alternative)\b", q) else "scenario"
    for intent, pattern in INTENT_RULES:
        if re.search(pattern, q):
            return intent
    return "snapshot"


def build_scenario(question: str, row) -> dict:
    """Turn free text into a concrete, simulation-ready scenario spec."""
    q = question.lower()
    spec: dict = {"label": "Your scenario"}

    amount = parse_amount(q)
    pct = parse_percentage(q)
    months = parse_months(q)

    if any(w in q for w in ("job loss", "lose my job", "lose my income", "unemployed", "no income")):
        spec.update(name="job_loss", label="Job loss", income_shock_pct=-1.0,
                    shock_duration_months=months or 4, recovery_factor=0.9)
    elif "salary cut" in q or ("income" in q and ("fall" in q or "drop" in q or "cut" in q or "reduc" in q)):
        spec.update(name="income_drop", label=f"Income {int((pct or 0.2)*100)}% lower",
                    income_change_pct=-(pct or 0.20))
    elif "inflation" in q:
        spec.update(name="inflation", label=f"Inflation +{int((pct or 0.04)*100)}pp",
                    inflation_rate=0.075 + (pct or 0.04))
    elif "medical" in q or "emergency" in q or "hospital" in q:
        spec.update(name="medical", label="Medical emergency",
                    lump_sum_expense=amount or 800_000, lump_sum_month=3)
    elif any(w in q for w in ("invest", "sip", "mutual fund", "portfolio")):
        spec.update(name="invest_more", label=f"Invest Rs {int(amount or 40_000):,}/month",
                    monthly_investment=amount or 40_000)
    elif amount and any(w in q for w in ("car", "vehicle", "house", "home", "flat", "apartment",
                                         "bike", "plot", "renovation", "wedding")):
        down = round(amount * 0.25, -3)
        spec.update(name="purchase", label=f"Purchase Rs {int(amount):,}",
                    lump_sum_expense=down, lump_sum_month=1,
                    new_debt_principal=round(amount - down, -3), new_debt_rate=0.20,
                    new_debt_term_months=60, new_debt_start_month=1)
    elif "debt" in q or "loan" in q:
        spec.update(name="debt_payoff", label="Aggressive debt payoff",
                    debt_payment_multiplier=3.0)
    elif "expense" in q or "spend" in q:
        spec.update(name="expense_change", label=f"Expenses {(pct or 0.1)*100:+.0f}%",
                    expense_change_pct=(pct or 0.10))
    else:
        spec.update(name="custom", label="Custom scenario")
    spec["horizon_months"] = 36
    return spec


# --------------------------------------------------------------------------------------
# node implementations
# --------------------------------------------------------------------------------------
def _record(state: dict, name: str, payload: dict) -> None:
    state.setdefault("tool_calls", []).append({"tool": name})
    state.setdefault("results", {})[name] = payload


def intent_node(state: dict) -> dict:
    q = state["question"]
    intent = detect_intent(q)
    state["intent"] = intent
    state["plan"] = PLANS.get(intent, ["analyst"])
    if intent == "scenario" or intent == "optimise":
        row = get_context().row(state["user_id"])
        state["scenario"] = build_scenario(q, row)
    return state


def research_node(state: dict) -> dict:
    hits = tools.knowledge_search(state["question"], k=4)
    state["sources"] = hits["hits"]
    _record(state, "knowledge_search", hits)
    return state


def analyst_node(state: dict) -> dict:
    uid = state["user_id"]
    _record(state, "get_user_snapshot", tools.get_user_snapshot(uid))
    _record(state, "calculate_cashflow", tools.calculate_cashflow(uid, months=12))
    return state


def forecast_node(state: dict) -> dict:
    _record(state, "forecast_expenses", tools.forecast_expenses(state["user_id"]))
    _record(state, "forecast_cashflow", tools.forecast_cashflow(state["user_id"]))
    return state


def risk_node(state: dict) -> dict:
    uid = state["user_id"]
    _record(state, "calculate_risk", tools.calculate_risk(uid))
    _record(state, "explain_risk", tools.explain_risk(uid))
    _record(state, "counterfactual_options", tools.counterfactual_options(uid))
    return state


def anomaly_node(state: dict) -> dict:
    _record(state, "detect_anomalies", tools.detect_anomalies(state["user_id"]))
    _record(state, "spending_intelligence", tools.spending_intelligence(state["user_id"]))
    return state


def goal_node(state: dict) -> dict:
    _record(state, "goal_outlook", tools.goal_outlook(state["user_id"]))
    return state


def scenario_node(state: dict) -> dict:
    uid = state["user_id"]
    spec = state.get("scenario", {"name": "custom"})
    _record(state, "compare_scenarios", tools.compare_scenarios(uid, [spec], n_paths=4000, horizon_months=36))
    _record(state, "stress_test", tools.stress_test(uid, horizon_months=36, n_paths=2500))
    return state


def decision_node(state: dict) -> dict:
    spec = state.get("scenario")
    if not spec:
        return state
    _record(state, "optimise_decision", tools.optimise_decision(
        state["user_id"], spec, n_paths=2500, horizon_months=36))
    return state


def dispatch_node(state: dict) -> dict:
    """Run the nodes selected by the intent router (keeps the trace honest)."""
    runners = {
        "research": research_node, "analyst": analyst_node, "forecast": forecast_node,
        "risk": risk_node, "anomaly": anomaly_node, "goal": goal_node,
        "scenario": scenario_node, "decision": decision_node,
    }
    for name in state.get("plan", []):
        fn = runners.get(name)
        if fn is None:
            continue
        t0 = __import__("time").time()
        try:
            state = fn(state) or state
            status = "ok"
        except Exception as exc:                        # noqa: BLE001
            status = f"error: {exc}"
            log.warning("agent node %s failed: %s", name, exc)
        state.setdefault("trace", []).append({
            "node": f"agent::{name}", "status": status,
            "ms": round((__import__("time").time() - t0) * 1000, 1)})
    return state


def router(state: dict) -> str:
    return "dispatch"


NODE_LABELS = {
    "agent::analyst": "Financial Analyst",
    "agent::risk": "Risk Analyst",
    "agent::scenario": "Scenario Analyst",
    "agent::decision": "Decision Agent",
    "agent::research": "Research Agent",
    "agent::anomaly": "Anomaly Analyst",
    "agent::goal": "Goal Analyst",
    "agent::forecast": "Forecasting Agent",
    "explanation": "Explanation Agent",
    "intent": "Intent Detection",
    "dispatch": "Agent Dispatcher",
}
