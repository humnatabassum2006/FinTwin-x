"""Asset / portfolio generator with correlated-ish price paths (GBM per asset class)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import settings
from common.logging_utils import get_logger
from common.utils import month_range
from data_generator.personas import PERSONAS

log = get_logger("gen.investments")

#          asset             drift   vol      min_ticket  liquidity(0-1)
ASSETS = {
    "kse100_equity":    dict(drift=0.13, vol=0.24, min=10_000, liquidity=0.85),
    "mutual_fund":      dict(drift=0.11, vol=0.13, min=5_000, liquidity=0.9),
    "gold":             dict(drift=0.09, vol=0.16, min=20_000, liquidity=0.8),
    "crypto":           dict(drift=0.16, vol=0.62, min=5_000, liquidity=0.7),
    "fixed_deposit":    dict(drift=0.145, vol=0.005, min=25_000, liquidity=0.5),
    "sukuk":            dict(drift=0.135, vol=0.02, min=25_000, liquidity=0.6),
    "prize_bonds":      dict(drift=0.07, vol=0.01, min=1_500, liquidity=0.95),
    "rental_property":  dict(drift=0.08, vol=0.09, min=500_000, liquidity=0.15),
}
ASSET_KEYS = list(ASSETS)

# risk-appetite conditioned allocation
ALLOCATION = {
    "conservative": dict(kse100_equity=0.10, mutual_fund=0.20, gold=0.15, fixed_deposit=0.35,
                         sukuk=0.15, prize_bonds=0.05),
    "balanced":     dict(kse100_equity=0.25, mutual_fund=0.25, gold=0.15, crypto=0.03,
                         fixed_deposit=0.20, sukuk=0.07, prize_bonds=0.05),
    "aggressive":   dict(kse100_equity=0.38, mutual_fund=0.22, gold=0.10, crypto=0.15,
                         fixed_deposit=0.08, sukuk=0.04, prize_bonds=0.03),
}


@dataclass
class InvestmentResult:
    investments: pd.DataFrame
    monthly_value: np.ndarray       # (n_users, n_months) portfolio value


def _price_path(asset: str, n_months: int, rng: np.random.Generator) -> np.ndarray:
    a = ASSETS[asset]
    dt = 1 / 12
    shocks = rng.normal((a["drift"] - 0.5 * a["vol"] ** 2) * dt, a["vol"] * np.sqrt(dt), size=n_months)
    path = 100 * np.exp(np.cumsum(shocks))
    return path


def generate_investments(users: pd.DataFrame, income_m: np.ndarray, expense_m: np.ndarray,
                         emi_m: np.ndarray, months: list, rng: np.random.Generator) -> InvestmentResult:
    n_users, n_months = income_m.shape
    as_of = settings.AS_OF
    prices = {a: _price_path(a, n_months + 6, rng) for a in ASSET_KEYS}

    rows = []
    value_matrix = np.zeros((n_users, n_months))
    uid = 1
    for u in range(n_users):
        persona = PERSONAS[users["persona"].iloc[u]]
        if rng.random() > persona.invest_prob:
            continue
        surplus = np.maximum(income_m[u] - expense_m[u] - emi_m[u], 0)
        monthly_invest = float(np.median(surplus)) * persona.invest_share
        if monthly_invest < 500:
            continue
        alloc = ALLOCATION.get(users["risk_preference"].iloc[u], ALLOCATION["balanced"])
        assets = list(alloc)
        p = np.array([alloc[a] for a in assets]); p = p / p.sum()
        chosen = rng.choice(assets, size=min(len(assets), 1 + int(rng.poisson(1.2))), replace=False, p=p)

        months_held = int(rng.integers(3, n_months))
        for asset in chosen:
            a = ASSETS[asset]
            contrib_share = alloc[asset] / sum(alloc[x] for x in chosen)
            invested = max(monthly_invest * months_held * contrib_share * rng.uniform(0.5, 1.4), a["min"])
            buy_month = n_months - months_held + int(rng.integers(0, 3))
            buy_month = int(np.clip(buy_month, 0, n_months - 1))
            purchase_price = float(prices[asset][buy_month])
            current_price = float(prices[asset][n_months - 1])
            qty = invested / purchase_price
            rows.append({
                "investment_id": uid,
                "user_id": u + 1,
                "asset": asset,
                "quantity": round(qty, 6),
                "purchase_price": round(purchase_price, 4),
                "current_price": round(current_price, 4),
                "purchase_date": months[buy_month],
                "as_of_date": as_of,
                "liquidity_score": a["liquidity"],
                "expected_return": a["drift"],
                "volatility": a["vol"],
            })
            uid += 1
            # mark-to-market across the window
            path = prices[asset][:n_months] * qty
            shifted = np.concatenate([np.zeros(buy_month), path[: n_months - buy_month]])
            value_matrix[u] += shifted

    inv = pd.DataFrame(rows)
    log.info("portfolio: %s holdings for %s users", f"{len(inv):,}",
             f"{inv['user_id'].nunique() if len(inv) else 0:,}")
    return InvestmentResult(investments=inv, monthly_value=value_matrix)
