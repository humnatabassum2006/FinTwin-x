"""Liability generator: loan book with amortisation, EMIs and remaining balances."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import settings
from common.logging_utils import get_logger
from common.utils import add_months, emi
from data_generator.personas import PERSONAS

log = get_logger("gen.debts")

LOAN_TYPES = {
    #            prob   principal x monthly income   term months    annual rate range
    "personal_loan": (0.30, (0.8, 3.0), (12, 36), (0.22, 0.34)),
    "car_loan":      (0.18, (3.0, 9.0), (36, 60), (0.14, 0.24)),
    "home_loan":     (0.12, (12.0, 40.0), (120, 240), (0.11, 0.19)),
    "credit_card":   (0.25, (0.2, 1.2), (6, 24), (0.32, 0.45)),
    "student_loan":  (0.08, (1.0, 5.0), (24, 84), (0.03, 0.10)),
    "gold_loan":     (0.07, (0.5, 2.5), (6, 24), (0.10, 0.18)),
}
LOAN_KEYS = list(LOAN_TYPES)
LOAN_PROBS = np.array([LOAN_TYPES[k][0] for k in LOAN_KEYS]); LOAN_PROBS = LOAN_PROBS / LOAN_PROBS.sum()


@dataclass
class DebtResult:
    debts: pd.DataFrame
    monthly_emi: np.ndarray        # (n_users, n_months)


def _amortised_balance(principal: float, rate: float, pay: float, months_paid: int) -> float:
    bal = principal
    r = rate / 12
    for _ in range(int(max(months_paid, 0))):
        interest = bal * r
        bal = bal + interest - pay
        if bal <= 0:
            return 0.0
    return bal


def generate_debts(users: pd.DataFrame, months: list, rng: np.random.Generator) -> DebtResult:
    n_users = len(users)
    n_months = len(months)
    as_of = settings.AS_OF
    rows = []
    emi_matrix = np.zeros((n_users, n_months))

    for u in range(n_users):
        persona = PERSONAS[users["persona"].iloc[u]]
        if rng.random() > persona.debt_prob:
            continue
        n_loans = 1 + int(rng.poisson(0.35))
        n_loans = min(n_loans, 3)
        income = float(users["monthly_income_base"].iloc[u])
        for _ in range(n_loans):
            ltype = LOAN_KEYS[int(rng.choice(len(LOAN_KEYS), p=LOAN_PROBS))]
            _, p_mult, term_rng, rate_rng = LOAN_TYPES[ltype]
            principal = income * rng.uniform(*p_mult) * persona.debt_dti / max(persona.debt_dti, 1e-9)
            principal = float(np.round(principal * rng.uniform(0.6, 1.4), -3))
            if principal < 20_000:
                continue
            term = int(rng.integers(*term_rng))
            rate = float(rng.uniform(*rate_rng))
            pay = emi(principal, rate, term)
            start = add_months(as_of, -int(rng.integers(1, 40)))
            months_paid = (as_of.year - start.year) * 12 + (as_of.month - start.month)
            if months_paid >= term:      # already closed
                continue
            remaining = max(_amortised_balance(principal, rate, pay, months_paid), 0.0)
            rows.append({
                "loan_id": len(rows) + 1,
                "user_id": u + 1,
                "loan_type": ltype,
                "principal": round(principal, 2),
                "interest_rate": round(rate, 4),
                "monthly_payment": round(pay, 2),
                "remaining_balance": round(remaining, 2),
                "start_date": start,
                "end_date": add_months(start, term),
                "term_months": term,
            })
            # EMI obligation across the observed window
            start_i = max(0, n_months - months_paid - 1)
            end_i = min(n_months, start_i + term)
            emi_matrix[u, start_i:end_i] += pay

    debts = pd.DataFrame(rows)
    log.info("loan book: %s active loans for %s users", f"{len(debts):,}", f"{debts['user_id'].nunique() if len(debts) else 0:,}")
    return DebtResult(debts=debts, monthly_emi=emi_matrix)
