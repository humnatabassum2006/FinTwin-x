"""User population generator: profiles, household context and persona assignment."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import settings
from data_generator.personas import PERSONA_KEYS, PERSONA_WEIGHTS, PERSONAS, Persona

INCOME_TYPES = ("salaried", "business", "freelance", "rental", "mixed")
EMPLOYMENT_TYPES = ("permanent", "contract", "self_employed", "gig", "business_owner")
GOALS = ("emergency_fund", "house_down_payment", "car", "education", "hajj_umrah", "wedding", "retirement", "business_capital")
RISK_PREFERENCES = ("conservative", "balanced", "aggressive")


def generate_users(n_users: int, rng: np.random.Generator) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (users_df, persona_key_array)."""
    persona_idx = rng.choice(len(PERSONA_KEYS), size=n_users, p=np.array(PERSONA_WEIGHTS) / sum(PERSONA_WEIGHTS))
    persona_keys = np.array(PERSONA_KEYS)[persona_idx]

    cities = rng.choice(settings.CITIES, size=n_users)
    cost_idx = np.array([settings.CITY_COST_INDEX[c] for c in cities])

    base_income = np.empty(n_users)
    for i, key in enumerate(PERSONA_KEYS):
        m = persona_keys == key
        k = int(m.sum())
        if k:
            pr = PERSONAS[key]
            base_income[m] = rng.lognormal(np.log(pr.income_mu), pr.income_sigma, size=k)

    base_income = np.round(base_income * (0.75 + 0.25 * cost_idx), -2)

    age = np.clip(rng.normal(34, 9, n_users), 21, 65).astype(int)
    household = np.clip(rng.poisson(1.6, n_users) + 1, 1, 8)

    income_type = np.where(
        persona_keys == "irregular_income",
        rng.choice(["freelance", "business"], size=n_users),
        rng.choice(INCOME_TYPES, size=n_users, p=[0.62, 0.14, 0.10, 0.05, 0.09]),
    )
    employment_type = np.where(
        income_type == "salaried",
        rng.choice(["permanent", "contract"], size=n_users, p=[0.78, 0.22]),
        np.where(income_type == "business", "business_owner",
                 np.where(income_type == "freelance", "gig", "self_employed")),
    )

    risk_pref = np.array([PERSONAS[k].risk_preference for k in persona_keys])

    users = pd.DataFrame({
        "user_id": np.arange(1, n_users + 1, dtype=np.int32),
        "persona": persona_keys,
        "age": age.astype(np.int16),
        "household_size": household.astype(np.int8),
        "city": cities,
        "city_cost_index": cost_idx.astype(np.float32),
        "income_type": income_type,
        "employment_type": employment_type,
        "financial_goal": rng.choice(GOALS, size=n_users),
        "risk_preference": risk_pref,
        "monthly_income_base": base_income.astype(np.float64),
        "created_at": pd.Timestamp(settings.AS_OF) - pd.to_timedelta(rng.integers(180, 1100, n_users), unit="D"),
    })
    return users, persona_keys
