"""
Expense domain model.

Holds the merchant catalogue, payment-method mix, category seasonality and the
per-category transaction cadence used by the transaction engine. Keeping these
in a dedicated module makes the "expenses" domain explicit (and easy to extend
with real merchant data later).
"""
from __future__ import annotations

import numpy as np

from common.config import settings

CATEGORIES = list(settings.EXPENSE_CATEGORIES)

# --------------------------------------------------------------------------------------
# Merchant catalogue (Pakistan-flavoured so the demo feels real)
# --------------------------------------------------------------------------------------
MERCHANTS: dict[str, list[str]] = {
    "Housing": ["Property Owner", "Estate Agent", "Rent Transfer", "Housing Society", "Zameen.com Rent"],
    "Food": ["Imtiaz Super Market", "Naheed Supermarket", "Metro Cash & Carry", "Chase Up", "Bin Hashim",
             "KFC", "McDonald's", "Local Dhaba", "Foodpanda", "Bread & Beyond"],
    "Transport": ["Careem", "InDrive", "Bykea", "PSO Pump", "Shell Pump", "Total Parco",
                  "Metro Bus Card", "Suzuki Service Center", "Toll Plaza", "Uber"],
    "Education": ["Beaconhouse School", "LUMS Fee Portal", "Aga Khan University", "Punjab Group",
                  "Coursera", "British Council", "College Fee Counter"],
    "Healthcare": ["Aga Khan Hospital", "Shaukat Khanum Lab", "Indus Hospital", "Dawaai.pk",
                   "Servaid Pharmacy", "Dental Studio", "Clinic Consultation"],
    "Entertainment": ["Netflix", "Cinepax", "Arena Pakistan", "Spotify", "PSL Tickets",
                      "Nishat Hotel Buffet", "Cafe Zouk", "YouTube Premium"],
    "Shopping": ["Daraz.pk", "Gul Ahmed", "Khaadi", "Sapphire", "Bata", "Outfitters",
                 "Alkaram Studio", "iShopping", "Electronics Mall"],
    "Utilities": ["K-Electric", "Sui Gas", "PTCL", "Jazz", "Zong", "Telenor", "WASA Water Bill",
                  "Internet Bill", "Society Maintenance"],
    "Subscriptions": ["Netflix", "Spotify", "Amazon Prime", "JazzCash Premium", "Google One",
                      "Adobe CC", "Notion", "ChatGPT Plus"],
    "Other": ["Meezan Bank Fee", "Tahir Trust Donation", "Sadqah", "Gift Shop", "Salon",
              "Laundry", "Post Office", "Bank Charges"],
}

# --------------------------------------------------------------------------------------
# Payment-method mix per category (probabilities over the list below)
# --------------------------------------------------------------------------------------
PAYMENT_METHODS = ("cash", "debit_card", "credit_card", "bank_transfer", "mobile_wallet")

PAYMENT_MIX: dict[str, list[float]] = {
    "Housing": [0.10, 0.05, 0.05, 0.65, 0.15],
    "Food": [0.42, 0.22, 0.08, 0.06, 0.22],
    "Transport": [0.50, 0.10, 0.05, 0.05, 0.30],
    "Education": [0.25, 0.20, 0.10, 0.35, 0.10],
    "Healthcare": [0.45, 0.20, 0.12, 0.13, 0.10],
    "Entertainment": [0.25, 0.20, 0.30, 0.05, 0.20],
    "Shopping": [0.20, 0.25, 0.30, 0.05, 0.20],
    "Utilities": [0.20, 0.10, 0.10, 0.25, 0.35],
    "Subscriptions": [0.02, 0.28, 0.55, 0.05, 0.10],
    "Other": [0.40, 0.18, 0.12, 0.15, 0.15],
}

# --------------------------------------------------------------------------------------
# Cadence: expected number of transactions per month per category
# --------------------------------------------------------------------------------------
TX_PER_MONTH: dict[str, float] = {
    "Housing": 1.0, "Food": 7.0, "Transport": 5.0, "Education": 0.35, "Healthcare": 0.45,
    "Entertainment": 1.8, "Shopping": 1.6, "Utilities": 1.8, "Subscriptions": 2.2, "Other": 1.0,
}

# Seasonality multiplier per calendar month (1-12). Pakistan-specific:
# Ramzan / Eid shopping peaks, summer electricity peaks, wedding season.
SEASONALITY = np.array([
    1.02, 0.94, 1.10, 1.16, 1.00, 1.12, 1.02, 0.97, 0.95, 1.03, 1.12, 1.09,
])

# Category-specific seasonality (rows = category, cols = calendar month 1..12)
CATEGORY_SEASONALITY = {
    "Food": [1.00, 0.95, 1.08, 1.12, 1.00, 1.05, 1.00, 0.98, 0.95, 1.00, 1.05, 1.06],
    "Utilities": [1.15, 1.10, 1.05, 1.10, 1.35, 1.45, 1.40, 1.30, 1.15, 0.95, 0.85, 1.05],
    "Shopping": [1.00, 0.90, 1.15, 1.25, 1.00, 1.15, 1.00, 1.05, 0.95, 1.05, 1.20, 1.30],
    "Entertainment": [1.00, 1.00, 1.05, 1.15, 1.10, 1.05, 1.00, 0.95, 0.95, 1.00, 1.10, 1.20],
    "Transport": [0.98, 0.95, 1.02, 1.05, 1.05, 1.08, 1.10, 1.05, 1.00, 1.00, 1.02, 1.06],
    "Education": [1.60, 0.60, 0.60, 1.00, 1.90, 0.70, 1.50, 0.80, 0.60, 0.90, 0.60, 0.70],
    "Healthcare": [1.10, 1.05, 1.00, 0.95, 0.95, 1.00, 1.05, 1.10, 1.05, 1.00, 1.00, 1.05],
    "Housing": [1.0] * 12,
    "Subscriptions": [1.0] * 12,
    "Other": [1.0] * 12,
}

# Typical day-of-month window for each category (rent on the 1st-5th, etc.)
DAY_WINDOW: dict[str, tuple[int, int]] = {
    "Housing": (1, 6), "Utilities": (5, 20), "Subscriptions": (1, 28), "Education": (1, 10),
    "Food": (1, 28), "Transport": (1, 28), "Healthcare": (1, 28), "Entertainment": (1, 28),
    "Shopping": (1, 28), "Other": (1, 28),
}

# Impulse / weekend behaviour
WEEKEND_UPLIFT = 1.28
LATE_NIGHT_SHARE = 0.09          # share of discretionary transactions after 23:00
IMPULSE_CATEGORIES = ("Shopping", "Entertainment", "Other", "Food")


def category_matrix() -> np.ndarray:
    """(n_categories x 12) seasonality matrix aligned with `CATEGORIES`."""
    return np.array([CATEGORY_SEASONALITY.get(c, [1.0] * 12) for c in CATEGORIES], dtype=np.float64)


def tx_rate_vector() -> np.ndarray:
    return np.array([TX_PER_MONTH[c] for c in CATEGORIES], dtype=np.float64)


# --------------------------------------------------------------------------------------
# flat catalogue + index (memory-efficient integer coding for millions of rows)
# --------------------------------------------------------------------------------------
def merchant_catalogue() -> tuple[list[str], list[list[int]]]:
    """Return (flat merchant names, per-category list of global indices)."""
    flat: list[str] = []
    index: list[list[int]] = []
    for c in CATEGORIES:
        ids = []
        for name in MERCHANTS[c]:
            if name not in flat:
                flat.append(name)
            ids.append(flat.index(name))
        index.append(ids)
    return flat, index
