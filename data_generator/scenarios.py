"""
Pre-built what-if scenarios.

These are the templates exposed by the Scenario Lab UI and by the AI agents.
A scenario is a pure *parameter delta* — the simulation engine (not the LLM)
does the maths.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class ScenarioTemplate:
    key: str
    label: str
    icon: str
    question: str
    description: str
    params: dict = field(default_factory=dict)


TEMPLATES: dict[str, ScenarioTemplate] = {
    "job_loss": ScenarioTemplate(
        key="job_loss", label="Job loss (4 months)", icon="briefcase",
        question="What if I lose my income for 4 months?",
        description="Income drops to zero for N months, then partially recovers.",
        params={"income_shock_pct": -1.0, "shock_duration_months": 4, "recovery_factor": 0.9},
    ),
    "salary_cut": ScenarioTemplate(
        key="salary_cut", label="Salary cut 20%", icon="trending-down",
        question="What happens if my income falls by 20%?",
        description="Permanent income reduction of 20% from month 1.",
        params={"income_change_pct": -0.20},
    ),
    "inflation_spike": ScenarioTemplate(
        key="inflation_spike", label="Inflation +4%", icon="flame",
        question="What if inflation rises by 4%?",
        description="Expenses grow 4 percentage points faster than baseline.",
        params={"inflation_rate": 0.13},
    ),
    "car_purchase": ScenarioTemplate(
        key="car_purchase", label="Buy a Rs 4M car", icon="car",
        question="Can I afford a Rs 4,000,000 car?",
        description="Rs 1M down payment, Rs 3M financed over 5 years at 20% p.a.",
        params={"lump_sum_expense": 1_000_000, "new_debt_principal": 3_000_000,
                "new_debt_rate": 0.20, "new_debt_term_months": 60},
    ),
    "invest_more": ScenarioTemplate(
        key="invest_more", label="Invest Rs 40k / month", icon="chart-line",
        question="What if I double my monthly investment?",
        description="Raise monthly investment from current level to Rs 40,000.",
        params={"monthly_investment": 40_000},
    ),
    "medical_emergency": ScenarioTemplate(
        key="medical_emergency", label="Medical emergency Rs 800k", icon="heartbeat",
        question="What if I face a Rs 800,000 medical emergency?",
        description="One-off shock expense, partially covered by emergency fund.",
        params={"lump_sum_expense": 800_000},
    ),
    "debt_payoff": ScenarioTemplate(
        key="debt_payoff", label="Aggressive debt payoff", icon="shield",
        question="What if I clear my debt 3x faster?",
        description="Triple the EMI until the loan book is cleared.",
        params={"debt_payment_multiplier": 3.0},
    ),
    "side_income": ScenarioTemplate(
        key="side_income", label="Side income Rs 50k", icon="plus-circle",
        question="What if I earn Rs 50,000 extra per month?",
        description="Additional recurring monthly income of Rs 50,000.",
        params={"extra_monthly_income": 50_000},
    ),
}


def list_templates() -> list[dict]:
    return [asdict(t) for t in TEMPLATES.values()]


def get_template(key: str) -> dict:
    return asdict(TEMPLATES[key])


def resolve(key: str, overrides: dict[str, Any] | None = None) -> dict:
    """Template defaults merged with user/agent overrides."""
    params = dict(TEMPLATES[key].params)
    params.update(overrides or {})
    return params
