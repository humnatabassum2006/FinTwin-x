"""
Financial personas (archetypes).

Each archetype is a *parameter bundle* for the synthetic life generator:
income process, expense structure, savings discipline, debt appetite,
investment behaviour and shock sensitivity. Sampling thousands of users from
these archetypes produces a population with realistic heterogeneity, which is
what the ML / simulation layers need in order to learn anything meaningful.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Persona:
    key: str
    label: str
    weight: float
    description: str

    # --- income process -------------------------------------------------------
    income_mu: float            # median monthly income (PKR) — lognormal location
    income_sigma: float         # lognormal spread across users
    income_growth: float        # annual salary growth
    income_vol: float           # month-to-month noise (lognormal sigma)
    irregular_prob: float       # probability of a bad / missing month
    irregular_severity: float   # income multiplier during a bad month

    # --- expense structure ----------------------------------------------------
    expense_ratio: float        # total expense / income target
    expense_vol: float
    shares: dict = field(default_factory=dict)   # category -> share of spend
    discretionary_bias: float = 1.0

    # --- behaviour ------------------------------------------------------------
    savings_rate: float = 0.10
    invest_share: float = 0.30      # share of surplus invested
    impulse_rate: float = 0.20      # impulse purchases per month
    cash_buffer_months: float = 3.0  # starting emergency fund

    # --- balance sheet --------------------------------------------------------
    debt_prob: float = 0.25
    debt_dti: float = 1.0           # loan principal as multiple of monthly income
    invest_prob: float = 0.50
    risk_preference: str = "balanced"

    # --- shocks ---------------------------------------------------------------
    shock_prob: float = 0.05        # monthly probability of an adverse event
    fragile: bool = False


_SHARES = {
    "conservative_saver": dict(Housing=0.26, Food=0.24, Transport=0.09, Utilities=0.09,
                               Education=0.06, Healthcare=0.06, Entertainment=0.04,
                               Shopping=0.09, Subscriptions=0.02, Other=0.05),
    "balanced_steady": dict(Housing=0.25, Food=0.23, Transport=0.10, Utilities=0.09,
                            Education=0.05, Healthcare=0.05, Entertainment=0.07,
                            Shopping=0.11, Subscriptions=0.02, Other=0.03),
    "overspender": dict(Housing=0.20, Food=0.20, Transport=0.09, Utilities=0.07,
                        Education=0.03, Healthcare=0.04, Entertainment=0.14,
                        Shopping=0.19, Subscriptions=0.02, Other=0.02),
    "aggressive_investor": dict(Housing=0.22, Food=0.20, Transport=0.09, Utilities=0.08,
                                Education=0.05, Healthcare=0.05, Entertainment=0.07,
                                Shopping=0.16, Subscriptions=0.03, Other=0.05),
    "debt_heavy": dict(Housing=0.24, Food=0.22, Transport=0.09, Utilities=0.09,
                       Education=0.06, Healthcare=0.06, Entertainment=0.05,
                       Shopping=0.12, Subscriptions=0.02, Other=0.05),
    "irregular_income": dict(Housing=0.22, Food=0.23, Transport=0.11, Utilities=0.09,
                             Education=0.05, Healthcare=0.05, Entertainment=0.07,
                             Shopping=0.13, Subscriptions=0.02, Other=0.03),
    "financially_fragile": dict(Housing=0.28, Food=0.27, Transport=0.11, Utilities=0.10,
                                Education=0.05, Healthcare=0.07, Entertainment=0.03,
                                Shopping=0.06, Subscriptions=0.01, Other=0.02),
    "young_aspirational": dict(Housing=0.27, Food=0.21, Transport=0.11, Utilities=0.08,
                               Education=0.10, Healthcare=0.03, Entertainment=0.09,
                               Shopping=0.08, Subscriptions=0.02, Other=0.01),
}

PERSONAS: dict[str, Persona] = {
    "conservative_saver": Persona(
        key="conservative_saver", label="Conservative Saver", weight=0.14,
        description="High savings discipline, low spending volatility, strong emergency reserve.",
        income_mu=185_000, income_sigma=0.35, income_growth=0.08, income_vol=0.04,
        irregular_prob=0.01, irregular_severity=0.75,
        expense_ratio=0.62, expense_vol=0.07, shares=_SHARES["conservative_saver"],
        savings_rate=0.32, invest_share=0.45, impulse_rate=0.08, cash_buffer_months=7.5,
        debt_prob=0.12, debt_dti=0.8, invest_prob=0.85, risk_preference="conservative",
        shock_prob=0.03,
    ),
    "balanced_steady": Persona(
        key="balanced_steady", label="Balanced Steady", weight=0.20,
        description="Stable income, moderate savings, controlled discretionary spending.",
        income_mu=150_000, income_sigma=0.40, income_growth=0.07, income_vol=0.05,
        irregular_prob=0.02, irregular_severity=0.70,
        expense_ratio=0.74, expense_vol=0.09, shares=_SHARES["balanced_steady"],
        savings_rate=0.20, invest_share=0.35, impulse_rate=0.18, cash_buffer_months=4.5,
        debt_prob=0.25, debt_dti=1.2, invest_prob=0.65, risk_preference="balanced",
        shock_prob=0.04,
    ),
    "overspender": Persona(
        key="overspender", label="Overspender", weight=0.15,
        description="High discretionary spending, impulse purchases, thin savings.",
        income_mu=165_000, income_sigma=0.42, income_growth=0.06, income_vol=0.06,
        irregular_prob=0.03, irregular_severity=0.70,
        expense_ratio=0.95, expense_vol=0.16, shares=_SHARES["overspender"],
        savings_rate=0.03, invest_share=0.10, impulse_rate=0.85, cash_buffer_months=1.4,
        debt_prob=0.42, debt_dti=1.6, invest_prob=0.35, risk_preference="aggressive",
        shock_prob=0.07,
    ),
    "aggressive_investor": Persona(
        key="aggressive_investor", label="Investor", weight=0.12,
        description="High investment allocation, accepts volatility, long-horizon goals.",
        income_mu=260_000, income_sigma=0.45, income_growth=0.09, income_vol=0.07,
        irregular_prob=0.03, irregular_severity=0.65,
        expense_ratio=0.66, expense_vol=0.11, shares=_SHARES["aggressive_investor"],
        savings_rate=0.30, invest_share=0.80, impulse_rate=0.22, cash_buffer_months=5.0,
        debt_prob=0.22, debt_dti=1.1, invest_prob=0.98, risk_preference="aggressive",
        shock_prob=0.04,
    ),
    "debt_heavy": Persona(
        key="debt_heavy", label="Debt-Heavy", weight=0.11,
        description="High debt-to-income, large EMI burden, weak liquidity.",
        income_mu=140_000, income_sigma=0.38, income_growth=0.05, income_vol=0.06,
        irregular_prob=0.05, irregular_severity=0.60,
        expense_ratio=0.86, expense_vol=0.12, shares=_SHARES["debt_heavy"],
        savings_rate=0.06, invest_share=0.12, impulse_rate=0.30, cash_buffer_months=1.1,
        debt_prob=0.95, debt_dti=3.4, invest_prob=0.30, risk_preference="balanced",
        shock_prob=0.09,
    ),
    "irregular_income": Persona(
        key="irregular_income", label="Irregular Income", weight=0.11,
        description="Freelancer / business owner: lumpy income, strong months and dry months.",
        income_mu=145_000, income_sigma=0.55, income_growth=0.04, income_vol=0.32,
        irregular_prob=0.18, irregular_severity=0.35,
        expense_ratio=0.80, expense_vol=0.18, shares=_SHARES["irregular_income"],
        savings_rate=0.14, invest_share=0.28, impulse_rate=0.28, cash_buffer_months=3.2,
        debt_prob=0.30, debt_dti=1.3, invest_prob=0.45, risk_preference="balanced",
        shock_prob=0.08,
    ),
    "financially_fragile": Persona(
        key="financially_fragile", label="Financially Fragile", weight=0.10,
        description="Little to no emergency reserve, expenses track income almost exactly.",
        income_mu=95_000, income_sigma=0.35, income_growth=0.04, income_vol=0.09,
        irregular_prob=0.09, irregular_severity=0.55,
        expense_ratio=0.97, expense_vol=0.13, shares=_SHARES["financially_fragile"],
        savings_rate=0.01, invest_share=0.05, impulse_rate=0.20, cash_buffer_months=0.4,
        debt_prob=0.55, debt_dti=2.1, invest_prob=0.15, risk_preference="conservative",
        shock_prob=0.12, fragile=True,
    ),
    "young_aspirational": Persona(
        key="young_aspirational", label="Young Aspirational", weight=0.07,
        description="Early career, education spend, fast-rising income, aggressive goals.",
        income_mu=105_000, income_sigma=0.38, income_growth=0.14, income_vol=0.08,
        irregular_prob=0.04, irregular_severity=0.65,
        expense_ratio=0.84, expense_vol=0.14, shares=_SHARES["young_aspirational"],
        savings_rate=0.12, invest_share=0.35, impulse_rate=0.45, cash_buffer_months=1.8,
        debt_prob=0.28, debt_dti=1.0, invest_prob=0.55, risk_preference="aggressive",
        shock_prob=0.06,
    ),
}

PERSONA_KEYS = tuple(PERSONAS)
PERSONA_WEIGHTS = tuple(p.weight for p in PERSONAS.values())


def get_persona(key: str) -> Persona:
    return PERSONAS[key]
