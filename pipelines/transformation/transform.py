"""
Transformation layer — turns validated raw tables into analytics-ready marts.

Main artefact: `monthly_panel`, a tidy (user × month) panel with income,
expense by category, EMI obligations, investment value, derived cash flow and a
running cash balance. Everything downstream (features, forecasting, simulation)
reads this panel — not the raw ledger.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from common.utils import history_months, write_parquet

log = get_logger("transform")


def _month_index(df: pd.DataFrame, col: str, months: list) -> pd.Series:
    ym = df[col].to_numpy().astype("datetime64[M]")
    lut = {np.datetime64(m): i for i, m in enumerate(months)}
    return pd.Series(ym, index=df.index).map(lut)


# --------------------------------------------------------------------------------------
# per-domain aggregations
# --------------------------------------------------------------------------------------
def tx_monthly(tx: pd.DataFrame, months: list) -> pd.DataFrame:
    tx = tx.copy()
    tx["ym"] = tx["timestamp"].to_numpy().astype("datetime64[M]")
    hour = tx["timestamp"].dt.hour.to_numpy()
    dow = tx["timestamp"].dt.dayofweek.to_numpy()
    cat = tx["category"].astype(str).to_numpy()
    amt = tx["amount"].to_numpy(dtype="float64")

    tx["_weekend_amt"] = np.where(np.isin(dow, [5, 6]), amt, 0.0)
    tx["_latenight_amt"] = np.where(hour >= 23, amt, 0.0)
    tx["_disc_amt"] = np.where(np.isin(cat, list(settings.DISCRETIONARY_CATEGORIES)), amt, 0.0)

    agg = tx.groupby(["user_id", "ym"], observed=True).agg(
        expense=("amount", "sum"),
        tx_count=("amount", "size"),
        tx_mean=("amount", "mean"),
        tx_max=("amount", "max"),
        tx_std=("amount", "std"),
        merchants=("merchant", "nunique"),
        anomaly_count=("is_anomaly", "sum"),
        weekend_spend=("_weekend_amt", "sum"),
        latenight_spend=("_latenight_amt", "sum"),
        discretionary_spend=("_disc_amt", "sum"),
    )

    cat = (tx.pivot_table(index=["user_id", "ym"], columns="category", values="amount",
                          aggfunc="sum", observed=True)
             .add_prefix("exp_"))
    out = agg.join(cat, how="left")
    for c in settings.EXPENSE_CATEGORIES:
        col = f"exp_{c}"
        if col not in out.columns:
            out[col] = 0.0
        out[col] = out[col].fillna(0.0)
    return out.reset_index()


def income_monthly(inc: pd.DataFrame, months: list) -> pd.DataFrame:
    inc = inc.copy()
    inc["ym"] = inc["date"].to_numpy().astype("datetime64[M]")
    out = inc.groupby(["user_id", "ym"], observed=True).agg(
        income=("amount", "sum"), income_count=("amount", "size")).reset_index()
    return out


def debt_monthly(debts: pd.DataFrame, months: list) -> pd.DataFrame:
    """Explode the loan book into (user, month) EMI + outstanding balance."""
    if debts.empty:
        return pd.DataFrame(columns=["user_id", "ym", "emi", "debt_balance"])
    months_arr = np.array([np.datetime64(m) for m in months], dtype="datetime64[M]")
    rows = []
    for rec in debts.to_dict("records"):
        start = np.datetime64(rec["start_date"], "M")
        end = np.datetime64(rec["end_date"], "M")
        mask = (months_arr >= start) & (months_arr <= end)
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            continue
        r = rec["interest_rate"] / 12
        t = np.arange(idx.size)
        if r < 1e-12:
            bal = np.maximum(rec["principal"] - rec["monthly_payment"] * t, 0)
        else:
            growth = (1 + r) ** t
            bal = np.maximum(rec["principal"] * growth - rec["monthly_payment"] * (growth - 1) / r, 0)
        rows.append(pd.DataFrame({
            "user_id": rec["user_id"],
            "ym": months_arr[idx],
            "emi": rec["monthly_payment"],
            "debt_balance": bal,
        }))
    out = pd.concat(rows, ignore_index=True)
    return out.groupby(["user_id", "ym"], observed=True).agg(
        emi=("emi", "sum"), debt_balance=("debt_balance", "sum")).reset_index()


def investment_monthly(inv: pd.DataFrame, prices: pd.DataFrame, months: list) -> pd.DataFrame:
    """Mark-to-market: quantity × price index, only for months after purchase."""
    if inv.empty:
        return pd.DataFrame(columns=["user_id", "ym", "invest_value", "liquid_invest_value"])
    prices = prices.copy()
    prices["ym"] = prices["date"].to_numpy().astype("datetime64[M]")
    months_arr = np.array([np.datetime64(m) for m in months], dtype="datetime64[M]")

    inv = inv.copy()
    inv["purchase_ym"] = inv["purchase_date"].to_numpy().astype("datetime64[M]")
    inv["_k"] = 1
    prices["_k"] = 1
    grid = inv[["user_id", "asset", "quantity", "purchase_ym", "liquidity_score", "_k"]].merge(
        prices[["asset", "ym", "price_index", "_k"]], on=["asset", "_k"], how="inner")
    grid = grid[grid["ym"] >= grid["purchase_ym"]]
    grid["value"] = grid["quantity"] * grid["price_index"]
    grid["liquid_value"] = grid["value"] * grid["liquidity_score"]
    return (grid.groupby(["user_id", "ym"], observed=True)
                .agg(invest_value=("value", "sum"), liquid_invest_value=("liquid_value", "sum"))
                .reset_index())


# --------------------------------------------------------------------------------------
# panel assembly
# --------------------------------------------------------------------------------------
def tx_monthly_chunked(tx: pd.DataFrame, months: list, users_per_chunk: int = 1000) -> pd.DataFrame:
    """Aggregate the ledger in user-blocks so peak memory stays flat on big ledgers."""
    uids = np.sort(tx["user_id"].unique())
    if len(uids) <= users_per_chunk:
        return tx_monthly(tx, months)
    parts = []
    for block in np.array_split(uids, int(np.ceil(len(uids) / users_per_chunk))):
        sub = tx[tx["user_id"].isin(block)]
        if len(sub):
            parts.append(tx_monthly(sub, months))
        del sub
    return pd.concat(parts, ignore_index=True)


def build_panel(users: pd.DataFrame, tx_m: pd.DataFrame, inc: pd.DataFrame, debts: pd.DataFrame,
                inv: pd.DataFrame, prices: pd.DataFrame, months: list) -> pd.DataFrame:
    """Assemble the (user × month) analytics panel from the domain aggregates."""
    t0 = time.time()
    inc_m = income_monthly(inc, months)
    debt_m = debt_monthly(debts, months)
    inv_m = investment_monthly(inv, prices, months)
    log.info("aggregations built in %.1fs", time.time() - t0)

    grid = pd.MultiIndex.from_product(
        [users["user_id"].to_numpy(), np.array([np.datetime64(m) for m in months], dtype="datetime64[M]")],
        names=["user_id", "ym"]).to_frame(index=False)

    panel = (grid.merge(inc_m, on=["user_id", "ym"], how="left")
                 .merge(tx_m, on=["user_id", "ym"], how="left")
                 .merge(debt_m, on=["user_id", "ym"], how="left")
                 .merge(inv_m, on=["user_id", "ym"], how="left"))
    fill = ["income", "expense", "emi", "debt_balance", "invest_value", "liquid_invest_value",
            "tx_count", "weekend_spend", "latenight_spend", "discretionary_spend",
            "anomaly_count", "merchants"] + [f"exp_{c}" for c in settings.EXPENSE_CATEGORIES]
    for c in fill:
        if c in panel.columns:
            panel[c] = panel[c].fillna(0.0)
    panel["tx_mean"] = panel["tx_mean"].fillna(panel["expense"])
    panel["tx_max"] = panel["tx_max"].fillna(panel["expense"])
    panel["tx_std"] = panel["tx_std"].fillna(0.0)

    # ---- derived cash flow ------------------------------------------------------
    panel["total_outflow"] = panel["expense"] + panel["emi"]
    panel["net_cash_flow"] = panel["income"] - panel["total_outflow"]
    panel["savings"] = panel["net_cash_flow"].clip(lower=0)

    # investment propensity: how much of the surplus this user historically deploys
    inv_intensity = (panel.groupby("user_id")["invest_value"].max() /
                     (panel.groupby("user_id")["income"].mean() * 12 + 1)).clip(0.05, 0.9)
    propensity = panel["user_id"].map(inv_intensity).fillna(0.25)
    panel["invest_flow"] = (panel["net_cash_flow"].clip(lower=0) * propensity).round(2)

    opening = users.set_index("user_id")["opening_cash_balance"]
    panel = panel.sort_values(["user_id", "ym"])
    net = (panel["net_cash_flow"] - panel["invest_flow"]).to_numpy()
    panel["cash_delta"] = net
    panel["cash_balance"] = (
        panel["user_id"].map(opening).to_numpy() +
        pd.Series(net).groupby(panel["user_id"].to_numpy()).cumsum().to_numpy()
    )
    panel["liquid_assets"] = panel["cash_balance"].clip(lower=0) + panel["liquid_invest_value"]
    panel["monthly_expense"] = panel["expense"]
    panel["year_month"] = pd.to_datetime(panel["ym"].to_numpy().astype("datetime64[D]"))
    panel = panel.drop(columns=["ym"])
    log.info("panel: %s rows (%s users × %s months) in %.1fs",
             f"{len(panel):,}", f"{panel['user_id'].nunique():,}", len(months), time.time() - t0)
    return panel


def clean_transactions(tx: pd.DataFrame) -> pd.DataFrame:
    """Type / range cleansing of the raw ledger (no full-frame string copies)."""
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").astype("float32")
    tx = tx[(tx["amount"] > 0) & tx["timestamp"].notna()]
    tx["timestamp"] = pd.to_datetime(tx["timestamp"])
    tx = tx[tx["timestamp"] <= pd.Timestamp(settings.AS_OF)]
    if tx["merchant"].dtype == object:
        tx["merchant"] = tx["merchant"].astype(str).str.strip().astype("category")
    if tx["category"].dtype == object:
        tx["category"] = pd.Categorical(tx["category"].astype(str),
                                        categories=list(settings.EXPENSE_CATEGORIES))
    return tx


def write_transactions_streaming(tx: pd.DataFrame, path, rows_per_group: int = 500_000) -> None:
    """Write a very large ledger to Parquet with bounded memory (row-group streaming)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = pa.Table.from_pandas(tx.head(1000), preserve_index=False).schema
    writer = pq.ParquetWriter(path, schema, compression="snappy")
    try:
        for start in range(0, len(tx), rows_per_group):
            chunk = tx.iloc[start:start + rows_per_group]
            writer.write_table(pa.Table.from_pandas(chunk, schema=schema, preserve_index=False))
            del chunk
    finally:
        writer.close()


def transform_transaction_ledger(tx: pd.DataFrame, months: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Clean + aggregate the ledger, then release it. Returns (clean_tx, tx_monthly)."""
    tx = clean_transactions(tx)
    agg = tx_monthly_chunked(tx, months)
    return tx, agg


def transform(frames: dict[str, pd.DataFrame], n_months: int = settings.HISTORY_MONTHS,
              tx_monthly: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    months = history_months(settings.AS_OF, n_months)
    users, inc = frames["users"], frames["income"]
    debts, inv, prices = frames["debts"], frames["investments"], frames["asset_prices"]
    tx = frames.get("transactions")

    if tx_monthly is None:
        if tx is None:
            raise ValueError("either transactions or tx_monthly must be provided")
        tx = clean_transactions(tx)
        tx_monthly = tx_monthly_chunked(tx, months)

    inc["amount"] = pd.to_numeric(inc["amount"], errors="coerce")
    inc = inc[inc["amount"] > 0].copy()

    for df in (users, debts, inv):
        for col in df.columns:
            if str(df[col].dtype) == "object" and col.endswith("date"):
                df[col] = pd.to_datetime(df[col], errors="coerce").dt.date

    panel = build_panel(users, tx_monthly, inc, debts, inv, prices, months)

    out = {
        "users": users,
        "income": inc,
        "debts": debts,
        "investments": inv,
        "financial_goals": frames["financial_goals"],
        "monthly_panel": panel,
    }
    for name, df in out.items():
        write_parquet(df, settings.PROCESSED_DIR / f"{name}.parquet")
    if tx is not None:
        write_transactions_streaming(tx, settings.PROCESSED_DIR / "transactions.parquet")
    log.info("processed marts written to %s", settings.PROCESSED_DIR)

    stats = {
        "rows": {k: int(len(v)) for k, v in out.items()},
        "panel_months": len(months),
        "window": [str(months[0]), str(months[-1])],
        "totals": {
            "income": float(panel["income"].sum()),
            "expense": float(panel["expense"].sum()),
            "emi": float(panel["emi"].sum()),
            "net_cash_flow": float(panel["net_cash_flow"].sum()),
        },
    }
    (settings.PROCESSED_DIR / "transformation_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return out
