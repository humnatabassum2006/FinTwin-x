"""
Transaction engine.

Generates the full transaction ledger in a *vectorised* way:

    for every (user, month, category) we build a "plan row" with a monthly
    budget, sample how many transactions realise that budget, then explode the
    plan into individual transactions using np.repeat + group-wise weights.

That produces millions of realistic rows in seconds instead of hours.

Ground-truth labels (`is_anomaly`, `anomaly_type`) are injected so the anomaly
detection model can be evaluated properly (precision@K / recall@K).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from common.config import settings
from common.logging_utils import get_logger
from common.utils import history_months
from data_generator import expenses as ex
from data_generator.personas import PERSONAS

log = get_logger("gen.tx")

CAT_CODES = {c: i for i, c in enumerate(ex.CATEGORIES)}
ANOMALY_TYPES = ["none", "card_fraud_burst", "medical_emergency", "big_ticket_purchase", "travel_spike"]


@dataclass
class TransactionResult:
    transactions: pd.DataFrame
    monthly_expense: np.ndarray           # (n_users, n_months)
    monthly_by_category: np.ndarray       # (n_users, n_months, n_categories)
    cash_trajectory: np.ndarray           # (n_users, n_months)
    opening_cash: np.ndarray              # (n_users,)


# --------------------------------------------------------------------------------------
# 1. monthly expense totals
# --------------------------------------------------------------------------------------
def monthly_expense_matrix(users: pd.DataFrame, income_m: np.ndarray, emi_m: np.ndarray,
                           rng: np.random.Generator, months: list) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate spending with a *behavioural feedback loop*.

    Households do not mechanically overspend into bankruptcy: when liquid
    buffers get thin they curtail discretionary spending. Modelling that loop is
    what makes the synthetic population behave like real people (and it is what
    the stress-testing engine later exploits).

    Returns (expense_matrix, cash_trajectory, opening_cash).
    """
    n_users, n_months = income_m.shape
    persona = users["persona"].to_numpy()
    ratio = np.array([PERSONAS[k].expense_ratio for k in persona])
    vol = np.array([PERSONAS[k].expense_vol for k in persona])
    fragile = np.array([PERSONAS[k].fragile for k in persona])
    buffer_months = np.array([PERSONAS[k].cash_buffer_months for k in persona])

    month_num = np.array([m.month for m in months])
    season = ex.SEASONALITY[month_num - 1]
    desired_ratio = ratio[:, None] * season[None, :] * 0.90

    # month-by-month simulation with adaptive curtailment
    expense = np.zeros((n_users, n_months))
    income_smooth = pd.DataFrame(income_m.T).rolling(3, min_periods=1).mean().to_numpy().T
    noise = rng.lognormal(-0.5 * vol[:, None] ** 2, vol[:, None], size=(n_users, n_months))
    target = np.where(fragile[:, None], ratio[:, None] * income_m, income_smooth) * \
        desired_ratio * noise

    opening = np.maximum(buffer_months * np.nanmedian(target, axis=1), 0.0) * rng.uniform(0.85, 1.35, n_users)
    cash = opening.copy()
    traj = np.zeros((n_users, n_months))
    for t in range(n_months):
        desired = np.maximum(target[:, t], 1500.0)
        cover = cash / np.maximum(desired, 1.0)
        # curtail spending progressively when less than ~1 month of cover remains
        factor = np.clip(0.70 + 0.30 * (cover / 1.0), 0.70, 1.0)
        factor = np.where(cover >= 1.0, 1.0, factor)
        expense[:, t] = np.maximum(desired * factor, 1200.0)
        cash = cash + income_m[:, t] - expense[:, t] - emi_m[:, t]
        traj[:, t] = cash
    return expense, traj, opening


# --------------------------------------------------------------------------------------
# 2. plan rows: (user, month, category) -> budget
# --------------------------------------------------------------------------------------
def build_plan(expense_m: np.ndarray, users: pd.DataFrame, months: list, rng: np.random.Generator) -> pd.DataFrame:
    n_users, n_months = expense_m.shape
    n_cat = len(ex.CATEGORIES)

    persona = users["persona"].to_numpy()
    share_mat = np.zeros((n_users, n_cat))
    for i, key in enumerate(PERSONAS):
        m = persona == key
        if m.any():
            v = np.array([PERSONAS[key].shares.get(c, 0.0) for c in ex.CATEGORIES])
            share_mat[m] = v / v.sum()
    # per-user jitter around the archetype shares (Dirichlet-ish)
    jitter = rng.lognormal(0, 0.22, size=share_mat.shape)
    share_mat = share_mat * jitter
    share_mat /= share_mat.sum(axis=1, keepdims=True)

    cat_season = ex.category_matrix()                    # (n_cat, 12)
    month_num = np.array([m.month for m in months])
    season_by_cm = cat_season[:, month_num - 1]          # (n_cat, n_months)

    # (n_users, n_months, n_cat) budget matrix
    budget = (expense_m[:, :, None] * share_mat[:, None, :] *
              np.transpose(season_by_cm, (1, 0))[None, :, :])
    budget = np.maximum(budget, 0.0)

    plan = pd.DataFrame({
        "u": np.repeat(np.arange(n_users), n_months * n_cat),
        "m": np.tile(np.repeat(np.arange(n_months), n_cat), n_users),
        "c": np.tile(np.arange(n_cat), n_users * n_months),
        "budget": budget.ravel(),
    })
    plan = plan[plan["budget"] > 200].reset_index(drop=True)
    return plan


# --------------------------------------------------------------------------------------
# 3. explode plan rows into individual transactions
# --------------------------------------------------------------------------------------
def expand_transactions(plan: pd.DataFrame, users: pd.DataFrame, months: list,
                        rng: np.random.Generator) -> pd.DataFrame:
    n_cat = len(ex.CATEGORIES)
    rates = ex.tx_rate_vector()[plan["c"].to_numpy()]
    persona_disc = np.array([PERSONAS[k].discretionary_bias for k in users["persona"].to_numpy()])
    disc_cat = np.isin(np.arange(n_cat), [CAT_CODES[c] for c in ex.CATEGORIES if c in
                                          ("Shopping", "Entertainment", "Other", "Subscriptions")])
    scale = np.where(disc_cat[plan["c"].to_numpy()], persona_disc[plan["u"].to_numpy()], 1.0)
    lam = np.maximum(rates * scale, 0.05)

    counts = rng.poisson(lam)
    counts = np.maximum(counts, 1)
    counts = np.where(plan["c"].to_numpy() == CAT_CODES["Housing"], 1, counts)  # rent is one payment
    total = int(counts.sum())

    u = np.repeat(plan["u"].to_numpy(dtype=np.int32), counts)
    m = np.repeat(plan["m"].to_numpy(dtype=np.int16), counts)
    c = np.repeat(plan["c"].to_numpy(dtype=np.int8), counts)

    # split each plan budget across its transactions using skewed weights
    w = rng.lognormal(0.0, 0.65, size=total)
    offsets = np.zeros(len(counts), dtype=np.int64)
    np.cumsum(counts[:-1], out=offsets[1:])
    group_sum = np.add.reduceat(w, offsets)
    amounts = (plan["budget"].to_numpy()[np.repeat(np.arange(len(counts)), counts)] *
               (w / np.repeat(group_sum, counts)))

    # ---------------- dates & times ------------------------------------------------
    months_arr = np.array([np.datetime64(m_) for m_ in months])
    month_start = months_arr[m]
    base_day = np.array([ex.DAY_WINDOW[ex.CATEGORIES[cc]][0] for cc in c])
    win = np.array([ex.DAY_WINDOW[ex.CATEGORIES[cc]][1] - ex.DAY_WINDOW[ex.CATEGORIES[cc]][0] for cc in c])
    day_off = rng.integers(0, win + 1, size=total)

    # housing / subscriptions recur on (almost) the same day each month
    fixed = np.isin(c, [CAT_CODES["Housing"], CAT_CODES["Subscriptions"]])
    per_user_day = rng.integers(1, 8, size=len(users))
    day_off = np.where(fixed, per_user_day[u] - 1, day_off)

    dates = month_start + (base_day - 1 + day_off).astype("timedelta64[D]")
    last = np.datetime64(settings.AS_OF)
    dates = np.minimum(dates, last)

    hour = np.where(
        np.isin(c, [CAT_CODES["Entertainment"], CAT_CODES["Shopping"], CAT_CODES["Food"], CAT_CODES["Other"]]),
        rng.choice(np.arange(24), size=total, p=_hour_profile()),
        rng.integers(8, 22, size=total),
    )
    minute = rng.integers(0, 60, size=total)
    second = rng.integers(0, 60, size=total)
    ts = dates + hour.astype("timedelta64[h]") + minute.astype("timedelta64[m]") + second.astype("timedelta64[s]")

    # ---------------- merchants & payment methods (integer coded) ------------------
    merchant_flat, merchant_index = ex.merchant_catalogue()
    merchant_codes = np.zeros(total, dtype=np.int16)
    method_codes = np.zeros(total, dtype=np.int8)
    for code, cat in enumerate(ex.CATEGORIES):
        sel = c == code
        if not sel.any():
            continue
        k = int(sel.sum())
        merchant_codes[sel] = rng.choice(np.array(merchant_index[code], dtype=np.int16), size=k)
        p = np.array(ex.PAYMENT_MIX[cat], dtype=float)
        p = p / p.sum()
        method_codes[sel] = rng.choice(np.arange(len(ex.PAYMENT_METHODS), dtype=np.int8), size=k, p=p)

    # weekend uplift on discretionary categories
    dow = ((dates.astype("datetime64[D]").astype(np.int64) + 3) % 7)      # 0 = Monday
    weekend = np.isin(dow, [5, 6])
    disc_row = np.isin(c, [CAT_CODES[x] for x in ex.IMPULSE_CATEGORIES])
    amounts = np.where(weekend & disc_row, amounts * rng.uniform(1.0, ex.WEEKEND_UPLIFT, size=total), amounts)

    amounts = np.maximum(np.round(amounts, 0), 50.0).astype(np.float32)

    df = pd.DataFrame({
        "transaction_id": np.arange(1, total + 1, dtype=np.int64),
        "user_id": (u + 1).astype(np.int32),
        "timestamp": ts,
        "amount": amounts,
        "category": pd.Categorical.from_codes(c.astype(np.int8), categories=ex.CATEGORIES),
        "merchant": pd.Categorical.from_codes(merchant_codes, categories=merchant_flat),
        "payment_method": pd.Categorical.from_codes(method_codes, categories=list(ex.PAYMENT_METHODS)),
        "transaction_type": pd.Categorical.from_codes(np.zeros(total, dtype=np.int8), categories=["debit", "credit"]),
        "is_anomaly": np.zeros(total, dtype=np.int8),
        "anomaly_type": pd.Categorical.from_codes(np.zeros(total, dtype=np.int8), categories=ANOMALY_TYPES),
    })
    return df


def _hour_profile() -> np.ndarray:
    p = np.array([0.4, 0.2, 0.1, 0.1, 0.2, 0.6, 1.2, 1.8, 2.0, 2.2, 2.4, 2.6,
                  3.0, 2.8, 2.4, 2.4, 2.6, 3.2, 4.0, 4.6, 4.2, 3.2, 2.0, 1.1])
    return p / p.sum()


# --------------------------------------------------------------------------------------
# 4. impulses, seasonal big-tickets and labelled anomalies
# --------------------------------------------------------------------------------------
def inject_behavioural_extras(tx: pd.DataFrame, users: pd.DataFrame, expense_m: np.ndarray,
                              months: list, rng: np.random.Generator) -> pd.DataFrame:
    n_users, n_months = expense_m.shape
    persona = users["persona"].to_numpy()
    impulse_rate = np.array([PERSONAS[k].impulse_rate for k in persona])

    # ---- impulse purchases (natural heavy tail, NOT labelled as fraud) -----------
    n_impulse = int(rng.poisson(impulse_rate.sum() * n_months * 0.9))
    iu = rng.integers(0, n_users, size=n_impulse)
    im = rng.integers(0, n_months, size=n_impulse)
    months_arr = np.array([np.datetime64(m_) for m_ in months])
    start = months_arr[im]
    day_off = rng.integers(0, 27, size=n_impulse)
    dates = np.minimum(start + day_off.astype("timedelta64[D]"), np.datetime64(settings.AS_OF))
    dow = ((dates.astype("datetime64[D]").astype(np.int64) + 3) % 7)
    late = rng.random(n_impulse) < ex.LATE_NIGHT_SHARE
    hour = np.where(late, rng.integers(22, 25, size=n_impulse) % 24, rng.integers(9, 23, size=n_impulse))
    ts = dates + hour.astype("timedelta64[h]") + rng.integers(0, 60, size=n_impulse).astype("timedelta64[m]")
    cats = rng.choice(list(ex.IMPULSE_CATEGORIES), size=n_impulse, p=[0.42, 0.24, 0.14, 0.20])
    amt = expense_m[iu, im] * rng.lognormal(-2.0, 0.70, size=n_impulse) * np.where(np.isin(dow, [5, 6]), 1.15, 1.0)
    merchant_flat, merchant_index = ex.merchant_catalogue()
    impulse_mcodes = np.array([rng.choice(np.array(merchant_index[ex.CATEGORIES.index(c)], dtype=np.int16))
                               for c in cats], dtype=np.int16)
    impulse_df = pd.DataFrame({
        "transaction_id": np.arange(len(tx) + 1, len(tx) + 1 + n_impulse, dtype=np.int64),
        "user_id": (iu + 1).astype(np.int32),
        "timestamp": ts,
        "amount": np.maximum(np.round(amt, 0), 200.0),
        "category": pd.Categorical(cats, categories=ex.CATEGORIES),
        "merchant": pd.Categorical.from_codes(impulse_mcodes, categories=merchant_flat),
        "payment_method": pd.Categorical.from_codes(
            rng.choice(np.arange(len(ex.PAYMENT_METHODS), dtype=np.int8), size=n_impulse,
                       p=[0.15, 0.25, 0.40, 0.03, 0.17]),
            categories=list(ex.PAYMENT_METHODS)),
        "transaction_type": pd.Categorical.from_codes(np.zeros(n_impulse, dtype=np.int8),
                                                      categories=["debit", "credit"]),
        "is_anomaly": np.zeros(n_impulse, dtype=np.int8),
        "anomaly_type": pd.Categorical.from_codes(np.zeros(n_impulse, dtype=np.int8),
                                                  categories=ANOMALY_TYPES),
    })

    # ---- labelled anomalies -------------------------------------------------------
    anomalies = _inject_anomalies(users, months, rng, start_id=len(tx) + 1 + n_impulse,
                                  expense_m=expense_m)

    out = pd.concat([tx, impulse_df, anomalies], ignore_index=True)
    del tx, impulse_df
    order = np.lexsort((out["timestamp"].to_numpy("datetime64[ns]").astype("int64"),
                        out["user_id"].to_numpy(np.int32)))
    out = out.take(order).reset_index(drop=True)
    del order
    out["transaction_id"] = np.arange(1, len(out) + 1, dtype=np.int64)
    return out


def _inject_anomalies(users: pd.DataFrame, months: list, rng: np.random.Generator,
                      start_id: int, expense_m: np.ndarray) -> pd.DataFrame:
    """Four families of labelled anomalies — ground truth for Model D."""
    n_users, n_months = expense_m.shape
    months_arr = np.array([np.datetime64(m) for m in months], dtype="datetime64[ns]")
    persona = users["persona"].to_numpy()
    shock_p = np.array([PERSONAS[k].shock_prob for k in persona])

    frames = []
    tid = start_id

    # A) card fraud burst: 2-6 rapid transactions at an unusual merchant
    n_fraud_users = int(rng.poisson(n_users * 0.05))
    fu = rng.choice(n_users, size=n_fraud_users, replace=False)
    for u in fu:
        m = int(rng.integers(1, n_months))
        k = int(rng.integers(2, 7))
        day = int(rng.integers(1, 28))
        base = min(months_arr[m] + np.timedelta64(day - 1, "D"), np.datetime64(settings.AS_OF))
        ts = base + rng.integers(0, 3, size=k).astype("timedelta64[h]") + rng.integers(0, 60, size=k).astype("timedelta64[m]")
        amt = expense_m[u, m] * rng.uniform(0.25, 1.1, size=k)
        frames.append(pd.DataFrame({
            "transaction_id": np.arange(tid, tid + k),
            "user_id": u + 1,
            "timestamp": ts,
            "amount": np.round(np.maximum(amt, 500), 0),
            "category": rng.choice(["Shopping", "Other", "Entertainment"], size=k),
            "merchant": rng.choice(["Unknown POS", "Online Intl Merchant", "Electronics Mall", "Daraz.pk"], size=k),
            "payment_method": rng.choice(["credit_card", "debit_card"], size=k),
            "transaction_type": "debit",
            "is_anomaly": 1,
            "anomaly_type": "card_fraud_burst",
        }))
        tid += k

    # B) medical emergency: one very large healthcare transaction
    n_med = int(rng.poisson(n_users * 0.06))
    mu = rng.choice(n_users, size=n_med, replace=True)
    m = rng.integers(0, n_months, size=n_med)
    ts = np.minimum(months_arr[m] + rng.integers(0, 27, size=n_med).astype("timedelta64[D]"),
                    np.datetime64(settings.AS_OF))
    amt = expense_m[mu, m] * rng.uniform(1.5, 6.0, size=n_med)
    frames.append(pd.DataFrame({
        "transaction_id": np.arange(tid, tid + n_med),
        "user_id": mu + 1,
        "timestamp": ts,
        "amount": np.round(amt, 0),
        "category": "Healthcare",
        "merchant": rng.choice(["Aga Khan Hospital", "Indus Hospital", "Shaukat Khanum Lab"], size=n_med),
        "payment_method": rng.choice(["cash", "credit_card", "bank_transfer"], size=n_med),
        "transaction_type": pd.Categorical.from_codes(np.zeros(n_med, dtype=np.int8),
                                                      categories=["debit", "credit"]),
        "is_anomaly": 1,
        "anomaly_type": "medical_emergency",
    }))
    tid += n_med

    # C) appliance / big-ticket purchase
    n_app = int(rng.poisson(n_users * 0.06))
    au = rng.choice(n_users, size=n_app, replace=True)
    m = rng.integers(0, n_months, size=n_app)
    ts = np.minimum(months_arr[m] + rng.integers(0, 27, size=n_app).astype("timedelta64[D]"),
                    np.datetime64(settings.AS_OF))
    amt = expense_m[au, m] * rng.uniform(1.2, 4.5, size=n_app)
    frames.append(pd.DataFrame({
        "transaction_id": np.arange(tid, tid + n_app),
        "user_id": au + 1,
        "timestamp": ts,
        "amount": np.round(amt, 0),
        "category": "Shopping",
        "merchant": rng.choice(["Electronics Mall", "Daraz.pk", "Samsung Brand Store"], size=n_app),
        "payment_method": rng.choice(["credit_card", "bank_transfer", "cash"], size=n_app),
        "transaction_type": pd.Categorical.from_codes(np.zeros(n_app, dtype=np.int8),
                                                      categories=["debit", "credit"]),
        "is_anomaly": 1,
        "anomaly_type": "big_ticket_purchase",
    }))
    tid += n_app

    # D) travel spike: burst of transport + entertainment in a 7-day window
    n_trav = int(rng.poisson(n_users * 0.03))
    tu = rng.choice(n_users, size=n_trav, replace=False)
    rows = []
    for u in tu:
        m = int(rng.integers(0, n_months))
        k = int(rng.integers(4, 11))
        start = np.minimum(months_arr[m] + np.timedelta64(int(rng.integers(0, 21)), "D"),
                           np.datetime64(settings.AS_OF))
        ts = start + rng.integers(0, 7, size=k).astype("timedelta64[D]")
        ts = np.minimum(ts, np.datetime64(settings.AS_OF))
        amt = expense_m[u, m] * rng.uniform(0.15, 0.8, size=k)
        cats = rng.choice(["Transport", "Entertainment", "Food"], size=k, p=[0.45, 0.35, 0.20])
        merchants = [rng.choice(ex.MERCHANTS[c]) for c in cats]
        rows.append(pd.DataFrame({
            "transaction_id": np.arange(tid, tid + k),
            "user_id": u + 1,
            "timestamp": ts,
            "amount": np.round(amt, 0),
            "category": cats,
            "merchant": merchants,
            "payment_method": rng.choice(["credit_card", "cash", "mobile_wallet"], size=k),
            "transaction_type": pd.Categorical.from_codes(np.zeros(k, dtype=np.int8),
                                                          categories=["debit", "credit"]),
            "is_anomaly": 1,
            "anomaly_type": "travel_spike",
        }))
        tid += k
    if rows:
        frames.append(pd.concat(rows, ignore_index=True))

    if not frames:
        return pd.DataFrame(columns=["transaction_id", "user_id", "timestamp", "amount", "category",
                                     "merchant", "payment_method", "transaction_type",
                                     "is_anomaly", "anomaly_type"])
    out = pd.concat(frames, ignore_index=True)
    out["anomaly_type"] = pd.Categorical(out["anomaly_type"], categories=ANOMALY_TYPES)
    out["transaction_type"] = pd.Categorical(out["transaction_type"], categories=["debit", "credit"])
    out["user_id"] = out["user_id"].astype(np.int32)
    out["category"] = pd.Categorical(out["category"], categories=ex.CATEGORIES)
    out["payment_method"] = pd.Categorical(out["payment_method"], categories=list(ex.PAYMENT_METHODS))
    out["is_anomaly"] = out["is_anomaly"].astype(np.int8)
    return out


# --------------------------------------------------------------------------------------
# orchestrator
# --------------------------------------------------------------------------------------
def generate_transactions(users: pd.DataFrame, income_m: np.ndarray, emi_m: np.ndarray,
                          rng: np.random.Generator, n_months: int) -> TransactionResult:
    months = history_months(settings.AS_OF, n_months)
    expense_m, cash_traj, opening = monthly_expense_matrix(users, income_m, emi_m, rng, months)
    plan = build_plan(expense_m, users, months, rng)
    log.info("transaction plan rows: %s", f"{len(plan):,}")
    tx = expand_transactions(plan, users, months, rng)
    log.info("base transactions: %s", f"{len(tx):,}")
    tx = inject_behavioural_extras(tx, users, expense_m, months, rng)
    log.info("transactions after impulses/anomalies: %s (anomalies=%s)",
             f"{len(tx):,}", f"{int(tx['is_anomaly'].sum()):,}")

    # monthly aggregation matrices (used by the feature store & panel builder)
    tx["ym"] = tx["timestamp"].to_numpy().astype("datetime64[M]")
    ym_index = {np.datetime64(m, "M"): i for i, m in enumerate(months)}
    mi = tx["ym"].map(ym_index).to_numpy()
    by_cat = np.zeros((len(users), n_months, len(ex.CATEGORIES)))
    np.add.at(by_cat, (tx["user_id"].to_numpy() - 1, mi, tx["category"].cat.codes.to_numpy()),
              tx["amount"].to_numpy())
    tx = tx.drop(columns=["ym"])
    return TransactionResult(transactions=tx, monthly_expense=expense_m,
                             monthly_by_category=by_cat, cash_trajectory=cash_traj,
                             opening_cash=opening)
