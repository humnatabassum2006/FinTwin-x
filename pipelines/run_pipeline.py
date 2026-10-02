"""
End-to-end pipeline runner.

    python -m pipelines.run_pipeline

    raw sources → ingestion → validation → transformation → feature store
                 → Financial DNA → segmentation → training dataset
"""
from __future__ import annotations

import argparse
import json
import gc
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from common import settings
from common.db import register_views
from common.logging_utils import get_logger
from common.utils import add_months, history_months, write_parquet
from pipelines.features.build_features import compute_dna, compute_features, segment
from pipelines.features.labels import build_training_dataset, OUTCOME_MONTHS
from pipelines.ingestion.ingest import ingest
from pipelines.transformation.transform import transform
from pipelines.validation.validate import validate

log = get_logger("pipeline")

TRAIN_CUTOFF = add_months(settings.AS_OF, -OUTCOME_MONTHS)


def load_raw(tables: tuple[str, ...]) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(settings.RAW_DIR / f"{t}.parquet") for t in tables}


def run(write: bool = True) -> dict:
    """Memory-aware orchestration:

    the transaction ledger (millions of rows) is validated, aggregated and
    released BEFORE the other marts are materialised, so peak RSS stays flat.
    """
    t0 = time.time()
    manifest = ingest()
    tables = tuple(manifest["tables"])
    small = tuple(t for t in tables if t != "transactions")

    validation_results: list = []
    quarantined: dict = {}

    # ---- stage 1: large ledger ---------------------------------------------------
    from pipelines.validation.validate import validate_table, ValidationReport
    from pipelines.transformation.transform import (transform_transaction_ledger,
                                                    write_transactions_streaming)

    months = history_months(settings.AS_OF, settings.HISTORY_MONTHS)
    tx = pd.read_parquet(settings.RAW_DIR / "transactions.parquet")
    res, tx = validate_table(tx, "transactions", {})
    validation_results.extend(res)
    quarantined["transactions"] = 0
    tx, tx_monthly = transform_transaction_ledger(tx, months)
    log.info("transaction ledger aggregated → %s user-months", f"{len(tx_monthly):,}")
    if write:
        write_transactions_streaming(tx, settings.PROCESSED_DIR / "transactions.parquet")
    del tx
    gc.collect()

    # ---- stage 2: the rest --------------------------------------------------------
    frames = load_raw(small)
    for t, df in frames.items():
        res, clean = validate_table(df, t, frames)
        validation_results.extend(res)
        quarantined[t] = int(len(df) - len(clean))
        frames[t] = clean
        log.info("validated %-16s rows=%s quarantined=%s", t, f"{len(df):,}", quarantined[t])

    report = ValidationReport(started_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                              results=validation_results, quarantined=quarantined)
    payload = report.summary()
    (settings.PROCESSED_DIR / "validation_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    validation = payload
    if not validation["ok"]:
        log.warning("validation failures: %s", validation["failures"][:5])

    marts = transform(frames, n_months=settings.HISTORY_MONTHS, tx_monthly=tx_monthly)
    users, panel = marts["users"], marts["monthly_panel"]
    goals, debts = marts["financial_goals"], marts["debts"]

    serve = compute_features(panel, users, goals, debts, settings.AS_OF, window=12)
    serve = compute_dna(serve)
    serve = segment(serve)

    train_feats = compute_features(panel, users, goals, debts, TRAIN_CUTOFF, window=12)
    train_feats = compute_dna(train_feats)
    train = build_training_dataset(panel, train_feats, TRAIN_CUTOFF)

    if write:
        write_parquet(serve, settings.PROCESSED_DIR / "financial_features.parquet")
        write_parquet(train, settings.PROCESSED_DIR / "training_dataset.parquet")
        con = register_views()
        con.close()
        (settings.EXPORT_DIR / "population.json").write_text(json.dumps({
            "as_of": str(settings.AS_OF),
            "n_users": int(len(serve)),
            "n_transactions": int(manifest["tables"]["transactions"]["rows"]),
            "train_cutoff": str(TRAIN_CUTOFF),
            "stress_rate": float(train["stress_label"].mean()),
            "segments": serve["segment_label"].value_counts().to_dict(),
            "median_income": float(serve["monthly_income"].median()),
            "median_expense": float(serve["monthly_expense"].median()),
            "mean_health": float(serve["dna_financial_health"].mean()),
        }, indent=2, default=str), encoding="utf-8")

    stats = {
        "runtime_seconds": round(time.time() - t0, 2),
        "validation": validation,
        "rows": {k: int(len(v)) for k, v in marts.items()},
        "rows_transactions": int(manifest["tables"]["transactions"]["rows"]),
        "features": int(serve.shape[1]),
        "serve_rows": int(len(serve)),
        "train_rows": int(len(train)),
        "stress_rate": float(train["stress_label"].mean()),
        "train_cutoff": str(TRAIN_CUTOFF),
        "segments": serve["segment_label"].value_counts().to_dict(),
    }
    (settings.PROCESSED_DIR / "pipeline_report.json").write_text(
        json.dumps(stats, indent=2, default=str), encoding="utf-8")
    log.info("pipeline complete in %.1fs", time.time() - t0)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    stats = run()
    if not args.quiet:
        print(json.dumps({k: v for k, v in stats.items() if k != "validation"}, indent=2, default=str))
        print("validation:", json.dumps(stats["validation"]["summary"] if "summary" in stats["validation"]
                                        else {k: stats["validation"][k] for k in ("ok", "passed", "failed")}))


if __name__ == "__main__":
    main()
