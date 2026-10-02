"""Small numeric / formatting / IO helpers shared across the platform."""
from __future__ import annotations

import hashlib
import math
from datetime import date, timedelta
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# money & numbers
# --------------------------------------------------------------------------------------
def money(x: float | int | None, currency: str = "PKR", decimals: int = 0) -> str:
    """Format a number as PKR with thousand separators (and compact form for big values)."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    v = float(x)
    sign = "-" if v < 0 else ""
    v = abs(v)
    if v >= 1e7:
        return f"{sign}Rs {v/1e7:.2f} Cr"
    if v >= 1e5:
        return f"{sign}Rs {v/1e5:.2f} L"
    return f"{sign}Rs {v:,.{decimals}f}"


def pct(x: float | None, decimals: int = 1) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "—"
    return f"{x * 100:.{decimals}f}%"


def clamp(x: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return float(min(hi, max(lo, x)))


def minmax_scale(s: pd.Series, lo: float = 0.0, hi: float = 100.0) -> pd.Series:
    """Robust min-max scaling on the 1st–99th percentile (outliers clipped)."""
    s = pd.Series(s, dtype="float64")
    low, high = s.quantile(0.01), s.quantile(0.99)
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-12:
        return pd.Series(np.full(len(s), (lo + hi) / 2), index=s.index)
    return ((s.clip(low, high) - low) / (high - low)) * (hi - lo) + lo


def safe_div(a, b, default: float = 0.0):
    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(b) < 1e-12, default, a / np.where(np.abs(b) < 1e-12, 1, b))
    return out


def emi(principal: float, annual_rate: float, months: int) -> float:
    """Standard amortised equated monthly instalment."""
    if months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r < 1e-12:
        return principal / months
    return principal * r * (1 + r) ** months / ((1 + r) ** months - 1)


def cagr_to_monthly(annual: float) -> float:
    return (1 + annual) ** (1 / 12) - 1


def percentile_bands(paths: np.ndarray, qs: Sequence[float] = (5, 25, 50, 75, 95)) -> dict:
    """Vectorised percentile summary over a (n_paths, n_steps) matrix."""
    out = {f"p{q}": np.percentile(paths, q, axis=0).tolist() for q in qs}
    out["mean"] = paths.mean(axis=0).tolist()
    return out


def cvar(paths: np.ndarray, level: float = 0.05) -> float:
    """Conditional Value at Risk (expected shortfall of the worst `level` tail)."""
    flat = np.asarray(paths).ravel()
    if flat.size == 0:
        return float("nan")
    cut = np.quantile(flat, level)
    tail = flat[flat <= cut]
    return float(tail.mean()) if tail.size else float(cut)


# --------------------------------------------------------------------------------------
# hashing / ids
# --------------------------------------------------------------------------------------
def stable_hash(*parts) -> str:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode())
    return h.hexdigest()[:16]


def make_id(prefix: str, *parts) -> str:
    return f"{prefix}_{stable_hash(*parts)}"


# --------------------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------------------
def month_range(start: date, end: date) -> list[date]:
    """First-of-month dates from `start` to `end` inclusive."""
    cur = date(start.year, start.month, 1)
    out = []
    while cur <= end:
        out.append(cur)
        cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return out


def add_months(d: date, n: int) -> date:
    total = d.month - 1 + n
    y = d.year + total // 12
    m = total % 12 + 1
    day = min(d.day, [31, 29 if (y % 4 == 0 and y % 100 != 0) or y % 400 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def month_index(series: pd.Series, origin: date | None = None) -> pd.Series:
    """Months since origin (used as a trend feature)."""
    origin = origin or date(2024, 9, 1)
    return (series.dt.year - origin.year) * 12 + (series.dt.month - origin.month)


# --------------------------------------------------------------------------------------
# io
# --------------------------------------------------------------------------------------
def write_parquet(df: pd.DataFrame, path, partition_by: Iterable[str] | None = None) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path = Path(path) if not isinstance(path, str) else Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if partition_by:
        pq.write_to_dataset(pa.Table.from_pandas(df, preserve_index=False), root_path=str(path),
                            partition_cols=list(partition_by), compression="snappy")
    else:
        df.to_parquet(path, index=False, compression="snappy")


def read_parquet(path) -> pd.DataFrame:
    return pd.read_parquet(path)


from pathlib import Path  # noqa: E402  (kept at bottom to avoid shadowing above)


def history_months(as_of: date, n_months: int) -> list[date]:
    """First-of-month dates for the `n_months` window ending in `as_of`'s month."""
    end = date(as_of.year, as_of.month, 1)
    total = end.year * 12 + (end.month - 1)
    start_i = total - (n_months - 1)
    return [date(i // 12, i % 12 + 1, 1) for i in range(start_i, total + 1)]
