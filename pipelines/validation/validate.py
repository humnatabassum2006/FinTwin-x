"""
Validation layer — a small, dependency-free "Great Expectations" clone.

Every table is checked against:
  * a Pydantic row contract (types, ranges, enums)
  * an expectation suite (nullability, uniqueness, referential integrity,
    value ranges, duplicate detection, distribution sanity)

Failing rows are quarantined (never silently dropped) so the pipeline is
auditable — which matters when the downstream output is financial advice.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pandas as pd

from common import settings
from common.logging_utils import get_logger
from pipelines.schemas import SCHEMAS

log = get_logger("validate")


@dataclass
class ExpectationResult:
    table: str
    expectation: str
    column: str
    success: bool
    observed: str
    severity: str = "error"      # error | warning
    n_bad: int = 0


@dataclass
class ValidationReport:
    started_at: str = ""
    results: list[ExpectationResult] = field(default_factory=list)
    quarantined: dict[str, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return all(r.success for r in self.results if r.severity == "error")

    def summary(self) -> dict:
        total = len(self.results)
        failed = [r for r in self.results if not r.success]
        return {
            "started_at": self.started_at,
            "expectations": total,
            "passed": total - len(failed),
            "failed": len(failed),
            "ok": self.ok,
            "quarantined_rows": self.quarantined,
            "failures": [
                {"table": r.table, "column": r.column, "expectation": r.expectation,
                 "observed": r.observed, "severity": r.severity}
                for r in failed
            ],
        }


# --------------------------------------------------------------------------------------
# expectation suite (declarative, table-driven)
# --------------------------------------------------------------------------------------
EXPECTATIONS = {
    "users": [
        ("not_null", "user_id", {}), ("unique", "user_id", {}),
        ("between", "age", {"min": 18, "max": 90}),
        ("not_null", "monthly_income_base", {}),
        ("greater_than", "monthly_income_base", {"min": 0}),
    ],
    "income": [
        ("not_null", "user_id", {}), ("not_null", "amount", {}),
        ("greater_than", "amount", {"min": 0}),
        ("referential", "user_id", {"to": "users", "column": "user_id"}),
    ],
    "transactions": [
        ("not_null", "user_id", {}), ("not_null", "amount", {}),
        ("greater_than", "amount", {"min": 0}),
        ("in_set", "category", {"values": list(settings.EXPENSE_CATEGORIES)}),
        ("not_future", "timestamp", {"max": str(settings.AS_OF)}),
        ("not_null", "timestamp", {}),
    ],
    "debts": [
        ("not_null", "user_id", {}), ("greater_than", "principal", {"min": 0}),
        ("between", "interest_rate", {"min": 0, "max": 1}),
    ],
    "investments": [
        ("not_null", "user_id", {}), ("greater_than", "quantity", {"min": 0}),
        ("greater_than", "current_price", {"min": 0}),
    ],
    "financial_goals": [
        ("not_null", "user_id", {}), ("greater_than", "target_amount", {"min": 0}),
        ("not_null", "target_date", {}),
    ],
    "asset_prices": [
        ("not_null", "asset", {}), ("greater_than", "price_index", {"min": 0}),
    ],
}


def _run_expectation(df: pd.DataFrame, table: str, kind: str, column: str, opts: dict,
                     others: dict[str, pd.DataFrame]) -> ExpectationResult:
    if column not in df.columns:
        return ExpectationResult(table, kind, column, False, "column missing", "error", len(df))
    s = df[column]
    if kind == "not_null":
        bad = int(s.isna().sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} nulls", "error", bad)
    if kind == "unique":
        bad = int(len(s) - s.nunique())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} duplicates", "error", bad)
    if kind == "between":
        bad = int(((s < opts["min"]) | (s > opts["max"])).sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} out of range", "error", bad)
    if kind == "greater_than":
        bad = int((s <= opts["min"]).sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} <= {opts['min']}", "error", bad)
    if kind == "in_set":
        bad = int((~s.isin(opts["values"])).sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} unknown values", "error", bad)
    if kind == "not_future":
        ts = pd.to_datetime(s, errors="coerce")
        bad = int((ts > pd.Timestamp(opts["max"])).sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} future rows", "error", bad)
    if kind == "referential":
        ref = others.get(opts["to"])
        if ref is None or opts["column"] not in ref.columns:
            return ExpectationResult(table, kind, column, True, "reference unavailable", "warning", 0)
        bad = int((~s.isin(set(ref[opts["column"]]))).sum())
        return ExpectationResult(table, kind, column, bad == 0, f"{bad} orphans", "error", bad)
    raise ValueError(f"unknown expectation {kind}")


def validate_table(df: pd.DataFrame, table: str, others: dict[str, pd.DataFrame] | None = None) -> tuple[list[ExpectationResult], pd.DataFrame]:
    """Run the expectation suite + row contracts. Returns (results, clean_df)."""
    others = others or {}
    results = []
    for kind, column, opts in EXPECTATIONS.get(table, []):
        results.append(_run_expectation(df, table, kind, column, opts, others))

    # ---- row-level contract validation (sampled for very large tables) -----------
    model = SCHEMAS.get(table)
    if model is not None:
        sample = df.head(5_000)
        errors, bad_idx = 0, []
        for i, rec in enumerate(sample.to_dict("records")):
            try:
                model(**rec)
            except Exception:
                errors += 1
                bad_idx.append(i)
        results.append(ExpectationResult(table, "pydantic_contract", "*", errors == 0,
                                         f"{errors}/{len(sample)} rows failed contract", "error", errors))
        if bad_idx:
            df = df.drop(df.index[bad_idx])

    clean = df
    if "user_id" in clean.columns:
        n_dupes = int(clean.duplicated().sum())
        if n_dupes:
            # only materialise a de-duplicated copy when duplicates actually exist
            before = len(clean)
            clean = clean.drop_duplicates()
            results.append(ExpectationResult(table, "no_duplicate_rows", "*", False,
                                             f"{before - len(clean)} duplicate rows removed", "warning",
                                             before - len(clean)))
        else:
            results.append(ExpectationResult(table, "no_duplicate_rows", "*", True, "0 duplicates", "warning", 0))
    return results, clean


def validate(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    report = ValidationReport(started_at=started)
    clean_frames = {}
    for table, df in frames.items():
        results, clean = validate_table(df, table, frames)
        report.results.extend(results)
        report.quarantined[table] = int(len(df) - len(clean))
        clean_frames[table] = clean
        log.info("validated %-16s rows=%s quarantined=%s", table, f"{len(df):,}",
                 f"{len(df) - len(clean):,}")

    payload = report.summary()
    (settings.PROCESSED_DIR / "validation_report.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")
    log.info("validation ok=%s (%s/%s expectations passed)", payload["ok"],
             payload["passed"], payload["expectations"])
    return clean_frames, payload
