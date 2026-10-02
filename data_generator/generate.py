"""
FinTwin Data Generator — CLI entrypoint.

    python -m data_generator.generate --users 10000 --months 24 --seed 42

Writes the synthetic financial ecosystem (raw domain tables + market data) to
`data/raw/`, plus a population summary to `data/synthetic/`.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from common.utils import history_months, write_parquet
from data_generator import debts as debts_mod
from data_generator import goals as goals_mod
from data_generator import income as income_mod
from data_generator import investments as inv_mod
from data_generator import transactions as tx_mod
from data_generator import users as users_mod

log = get_logger("gen")

RAW_TABLES = ("users", "income", "transactions", "debts", "investments", "financial_goals", "asset_prices")


def build_population(n_users: int, n_months: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    months = history_months(settings.AS_OF, n_months)

    log.info("generating %s users across %s months (seed=%s)", f"{n_users:,}", n_months, seed)
    users, _ = users_mod.generate_users(n_users, rng)
    inc = income_mod.generate_income(users, rng, n_months)
    debt = debts_mod.generate_debts(users, months, rng)
    tx = tx_mod.generate_transactions(users, inc.monthly, debt.monthly_emi, rng, n_months)
    inv = inv_mod.generate_investments(users, inc.monthly, tx.monthly_expense, debt.monthly_emi, months, rng)
    goals = goals_mod.generate_goals(users, inc.monthly, tx.monthly_expense, debt.monthly_emi, rng)

    # opening cash balance (= accumulated emergency reserve at onboarding)
    from data_generator.personas import PERSONAS
    buffer_months = np.array([PERSONAS[k].cash_buffer_months for k in users["persona"].to_numpy()])
    med_expense = np.median(tx.monthly_expense, axis=1)
    users = users.assign(
        opening_cash_balance=np.round(buffer_months * med_expense * rng.uniform(0.85, 1.35, len(users)), -2),
        monthly_expense_estimate=np.round(med_expense, -2),
    )

    return {
        "users": users,
        "income": inc.income,
        "transactions": tx.transactions,
        "debts": debt.debts,
        "investments": inv.investments,
        "financial_goals": goals,
        "asset_prices": _asset_price_table(inv, months, rng),
        "months": months,
        "shocks": inc.shocks,
    }


def _asset_price_table(inv, months: list, rng: np.random.Generator) -> pd.DataFrame:
    """Monthly price index per asset class (market data feed)."""
    from data_generator.investments import ASSETS

    rows = []
    n = len(months) + 6
    for asset, spec in ASSETS.items():
        dt = 1 / 12
        shocks = rng.normal((spec["drift"] - 0.5 * spec["vol"] ** 2) * dt,
                            spec["vol"] * np.sqrt(dt), size=n)
        path = 100 * np.exp(np.cumsum(shocks))
        rows.append(pd.DataFrame({
            "date": months + [_future_month(months[-1], i) for i in range(1, 7)],
            "asset": asset,
            "price_index": np.round(path, 4),
        }))
    return pd.concat(rows, ignore_index=True)


def _future_month(d, k: int):
    total = d.month - 1 + k
    y = d.year + total // 12
    m = total % 12 + 1
    from datetime import date
    return date(y, m, 1)


def write_outputs(pop: dict, out_dir: Path, write_csv_sample: bool = True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {}
    for t in RAW_TABLES:
        df = pop[t]
        write_parquet(df, out_dir / f"{t}.parquet")
        stats[t] = int(len(df))
        log.info("wrote %-16s %10s rows", t, f"{len(df):,}")

    if write_csv_sample:
        sample_dir = Path(settings.SYNTHETIC_DIR) / "csv_sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        ids = np.sort(pop["users"]["user_id"].to_numpy()[:50])
        for t in ("users", "transactions", "income", "debts", "investments", "financial_goals"):
            df = pop[t]
            if "user_id" in df.columns:
                df = df[df["user_id"].isin(ids)]
            df.head(20_000).to_csv(sample_dir / f"{t}.csv", index=False)

    summary = summarise(pop, stats)
    (Path(settings.SYNTHETIC_DIR) / "population_summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def summarise(pop: dict, stats: dict) -> dict:
    users = pop["users"]
    tx = pop["transactions"]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "as_of": str(settings.AS_OF),
        "tables": stats,
        "persona_mix": users["persona"].value_counts().to_dict(),
        "city_mix": users["city"].value_counts().to_dict(),
        "median_income": float(users["monthly_income_base"].median()),
        "total_transaction_value": float(tx["amount"].sum()),
        "anomaly_rate": float(tx["is_anomaly"].mean()),
        "anomaly_breakdown": tx.loc[tx["is_anomaly"] == 1, "anomaly_type"].value_counts().to_dict(),
        "category_mix": (tx.groupby("category", observed=True)["amount"].sum() /
                         tx["amount"].sum()).round(4).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="FinTwin-X synthetic data generator")
    ap.add_argument("--users", type=int, default=10_000)
    ap.add_argument("--months", type=int, default=settings.HISTORY_MONTHS)
    ap.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    ap.add_argument("--out", type=str, default=str(settings.RAW_DIR))
    ap.add_argument("--no-csv", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    pop = build_population(args.users, args.months, args.seed)
    summary = write_outputs(pop, Path(args.out), write_csv_sample=not args.no_csv)
    summary["runtime_seconds"] = round(time.time() - t0, 2)
    log.info("done in %.1fs → %s", time.time() - t0, args.out)
    print(json.dumps({k: summary[k] for k in ("tables", "anomaly_rate", "runtime_seconds")}, indent=2, default=str))


if __name__ == "__main__":
    main()
