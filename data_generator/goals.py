"""Financial-goal generator: targets, deadlines and contribution behaviour."""
from __future__ import annotations

import numpy as np
import pandas as pd

from common.config import settings
from common.logging_utils import get_logger
from data_generator.personas import PERSONAS

log = get_logger("gen.goals")

GOAL_SPECS = {
    #                    target range (PKR)          horizon months   typical persona bias
    "emergency_fund":    ((150_000, 1_200_000), (6, 24)),
    "house_down_payment": ((1_500_000, 12_000_000), (24, 96)),
    "car":               ((1_200_000, 6_000_000), (12, 48)),
    "education":         ((500_000, 5_000_000), (12, 72)),
    "hajj_umrah":        ((600_000, 2_500_000), (6, 36)),
    "wedding":           ((800_000, 6_000_000), (12, 60)),
    "retirement":        ((5_000_000, 60_000_000), (60, 240)),
    "business_capital":  ((500_000, 8_000_000), (12, 60)),
}


def generate_goals(users: pd.DataFrame, income_m: np.ndarray, expense_m: np.ndarray,
                   emi_m: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    as_of = settings.AS_OF
    rows = []
    for u in range(len(users)):
        persona = PERSONAS[users["persona"].iloc[u]]
        primary = users["financial_goal"].iloc[u]
        n_goals = 1 + int(rng.poisson(0.55))
        n_goals = min(n_goals, 3)
        chosen = [primary]
        pool = [g for g in GOAL_SPECS if g != primary]
        chosen += list(rng.choice(pool, size=max(0, n_goals - 1), replace=False))

        surplus = np.median(np.maximum(income_m[u] - expense_m[u] - emi_m[u], 0))
        for priority, goal in enumerate(chosen, start=1):
            (tlo, thi), (hlo, hhi) = GOAL_SPECS[goal]
            income = float(users["monthly_income_base"].iloc[u])
            scale = np.clip(income / 150_000, 0.45, 3.2)
            target = float(np.round(rng.uniform(tlo, thi) * scale, -3))
            horizon = int(rng.integers(hlo, hhi))
            target_date = pd.Timestamp(as_of) + pd.DateOffset(months=horizon)

            # contribution discipline: persona driven, with real-world noise
            required = target / max(horizon, 1)
            discipline = np.clip(persona.savings_rate / 0.25, 0.2, 1.6) * rng.uniform(0.55, 1.25)
            contribution = float(np.round(min(required * discipline, max(surplus * 0.8, 0)), -2))

            elapsed = int(rng.integers(1, max(2, horizon // 2)))
            saved = contribution * elapsed * rng.uniform(0.8, 1.15)
            rows.append({
                "goal_id": len(rows) + 1,
                "user_id": u + 1,
                "goal_name": goal,
                "target_amount": round(target, 2),
                "current_amount": round(float(np.clip(saved, 0, target * 1.1)), 2),
                "target_date": target_date.date(),
                "monthly_contribution": round(contribution, 2),
                "priority": np.int8(priority),
                "months_elapsed": elapsed,
                "status": "active",
            })
    goals = pd.DataFrame(rows)
    log.info("goals: %s goals across %s users", f"{len(goals):,}", f"{goals['user_id'].nunique():,}")
    return goals
