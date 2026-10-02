"""
Storage layer.

DuckDB is the default analytical engine (zero-config, reads Parquet in place).
PostgreSQL + pgvector is supported for the transactional/serving layer and for
vector search: set DATABASE_URL and the same code path switches over.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from common.config import settings
from common.logging_utils import get_logger

log = get_logger("db")

TABLES = (
    "users", "income", "transactions", "debts", "investments",
    "financial_goals", "monthly_panel", "financial_features",
)


def duckdb_conn(read_only: bool = False):
    return duckdb.connect(str(settings.duckdb_path), read_only=read_only)


def register_views(con=None, base: Path | None = None) -> duckdb.DuckDBPyConnection:
    """Create/replace DuckDB views pointing at the processed Parquet files."""
    con = con or duckdb_conn()
    base = Path(base or settings.PROCESSED_DIR)
    for t in TABLES:
        p = base / f"{t}.parquet"
        if p.exists():
            con.execute(f"CREATE OR REPLACE VIEW {t} AS SELECT * FROM read_parquet('{p.as_posix()}')")
    return con


def query(sql: str, params: list | None = None) -> "duckdb.DuckDBPyRelation":
    con = duckdb_conn(read_only=True)
    try:
        rel = con.execute(sql, params or [])
        try:
            return rel.fetchdf()
        except Exception:
            return rel
    finally:
        con.close()


def pg_available() -> bool:
    return bool(settings.DATABASE_URL)


def pg_conn():  # pragma: no cover - requires a live database
    """Return a psycopg connection when DATABASE_URL is configured."""
    if not pg_available():
        raise RuntimeError("DATABASE_URL is not configured")
    import psycopg  # type: ignore

    return psycopg.connect(settings.DATABASE_URL)


def table_stats() -> dict:
    """Row counts for processed tables without mutating the DuckDB catalog.

    Health checks can arrive concurrently.  Creating/replacing persistent DuckDB
    views from every request causes catalog write/write conflicts, so counts are
    read directly from Parquet metadata instead.  This is read-only, fast and
    safe across concurrent requests and multiple worker threads.
    """
    import pyarrow.parquet as pq

    out: dict[str, int] = {}
    for t in TABLES:
        p = settings.PROCESSED_DIR / f"{t}.parquet"
        if not p.exists():
            out[t] = 0
            continue
        try:
            out[t] = int(pq.ParquetFile(p).metadata.num_rows)
        except Exception as exc:  # health should degrade, never crash
            log.warning("could not read parquet metadata for %s: %s", t, exc)
            out[t] = 0
    return out
