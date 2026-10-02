"""
Income process.

Salaried users: monthly salary on a fixed day + annual increment (July) +
Eid bonus (twice a year) + occasional overtime/freelance top-ups.
Business / freelance: AR(1) log-income with fat tails and dry months.

The generator also injects *income shocks* (job loss, salary cut, delayed
client payment). These drive the supervised label for the risk model.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import settings
from common.logging_utils import get_logger
from common.utils import history_months
from data_generator.personas import PERSONAS

log = get_logger("gen.income")

SOURCES = {
    "salaried": ["Employer Payroll", "Employer Payroll", "Bonus", "Overtime", "Freelance Side"],
    "business": ["Business Revenue", "Business Revenue", "Client Payment", "Rental Income"],
    "freelance": ["Client Payment", "Client Payment", "Upwork Payout", "Royalty"],
    "rental": ["Rental Income", "Rental Income", "Dividend", "Pension"],
    "mixed": ["Employer Payroll", "Business Revenue", "Rental Income", "Dividend"],
}


@dataclass
class IncomeResult:
    income: pd.DataFrame          # income ledger
    monthly: np.ndarray           # (n_users, n_months) matrix of realised income
    shocks: np.ndarray            # (n_users, n_months) int8: 0 none, 1 salary cut, 2 job loss, 3 delayed payment


def generate_income(users: pd.DataFrame, rng: np.random.Generator, n_months: int) -> IncomeResult:
    n_users = len(users)
    months = history_months(settings.AS_OF, n_months)
    m_idx = np.arange(n_months)

    persona_keys = users["persona"].to_numpy()
    base = users["monthly_income_base"].to_numpy()
    growth_annual = np.array([PERSONAS[k].income_growth for k in persona_keys])
    vol = np.array([PERSONAS[k].income_vol for k in persona_keys])
    irr_prob = np.array([PERSONAS[k].irregular_prob for k in persona_keys])
    irr_sev = np.array([PERSONAS[k].irregular_severity for k in persona_keys])
    shock_prob = np.array([PERSONAS[k].shock_prob for k in persona_keys])
    fragile = np.array([PERSONAS[k].fragile for k in persona_keys])

    # trend: annual increment applied from the July of each simulated year
    growth_monthly = (1 + growth_annual) ** (1 / 12) - 1
    trend = (1 + growth_monthly[:, None]) ** m_idx[None, :]

    noise = rng.lognormal(-0.5 * vol[:, None] ** 2, vol[:, None], size=(n_users, n_months))
    income_m = base[:, None] * trend * noise

    # --- irregular / dry months ------------------------------------------------
    bad = rng.random((n_users, n_months)) < irr_prob[:, None]
    sev = irr_sev[:, None] * rng.uniform(0.6, 1.1, size=(n_users, n_months))
    income_m = np.where(bad, income_m * sev, income_m)

    # --- seasonal bonuses ------------------------------------------------------
    month_num = np.array([m.month for m in months])
    is_eid = np.isin(month_num, [4, 6, 11])           # Eid-ul-Fitr / Eid-ul-Adha / winter
    income_m += np.where(is_eid[None, :], base[:, None] * rng.uniform(0.25, 0.85, size=(n_users, 3)).mean(axis=1)[:, None], 0.0)

    # --- injected shocks (labelled) -------------------------------------------
    shocks = np.zeros((n_users, n_months), dtype=np.int8)
    p = shock_prob[:, None] / 3.0
    p = np.clip(p + np.where(fragile[:, None], 0.02, 0.0), 0, 0.25)
    draws = rng.random((n_users, n_months))
    kinds = rng.integers(1, 4, size=(n_users, n_months), dtype=np.int8)
    hit = draws < p
    # job loss lasts 2-5 months, salary cut fades over 6 months, delayed payment = 1 month
    for u in np.flatnonzero(hit.any(axis=1)):
        for m in np.flatnonzero(hit[u]):
            kind = int(kinds[u, m])
            shocks[u, m] = kind
            if kind == 2:  # job loss
                dur = int(rng.integers(2, 6))
                end = min(n_months, m + dur)
                shocks[u, m:end] = 2
                income_m[u, m:end] *= rng.uniform(0.0, 0.15)
                # partial recovery afterwards
                if end < n_months:
                    income_m[u, end:] *= rng.uniform(0.85, 1.05)
            elif kind == 1:  # salary cut
                dur = int(rng.integers(4, 9))
                end = min(n_months, m + dur)
                shocks[u, m:end] = 1
                income_m[u, m:end] *= rng.uniform(0.65, 0.85)
            else:  # delayed payment
                income_m[u, m] *= rng.uniform(0.15, 0.5)
                if m + 1 < n_months:
                    income_m[u, m + 1] += (base[u] - income_m[u, m]) * rng.uniform(0.5, 0.95)

    income_m = np.maximum(income_m, 0.0).round(-1)

    # --- ledger ------------------------------------------------------------------
    rows_user, rows_month = np.nonzero(income_m > 0)
    amounts = income_m[rows_user, rows_month]
    itypes = users["income_type"].to_numpy()[rows_user]

    # primary monthly credit + occasional secondary credit
    n_rows = len(amounts)
    months_arr = np.array(months, dtype="datetime64[ns]")
    day = np.where(itypes == "salaried", rng.integers(1, 4, n_rows), rng.integers(1, 29, n_rows))
    m_start = months_arr[rows_month]
    dates = m_start + (day - 1).astype("timedelta64[D]")
    dates = np.minimum(dates, np.datetime64(settings.AS_OF))

    src_pool = np.array([rng.choice(SOURCES[t]) for t in itypes])
    recurring = np.where(itypes == "salaried", True, False)

    income_df = pd.DataFrame({
        "income_id": np.arange(1, n_rows + 1, dtype=np.int64),
        "user_id": (rows_user + 1).astype(np.int32),
        "date": dates,
        "amount": amounts.astype(np.float64),
        "source": src_pool,
        "income_type": itypes,
        "recurring": recurring,
    })
    log.info("income ledger: %s rows for %s users over %s months", f"{n_rows:,}", f"{n_users:,}", n_months)
    return IncomeResult(income=income_df, monthly=income_m, shocks=shocks)
