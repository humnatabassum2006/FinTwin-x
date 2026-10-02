"""
Ingestion layer.

Reads raw sources (Parquet / CSV / JSON exports / bank-style CSV), lands them in
DuckDB as `raw_*` tables and emits an ingestion manifest.

Design note: ingestion never mutates data — it only *lands* it. Every
transformation happens downstream so a failed run is always replayable.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd

from common import settings
from common.db import duckdb_conn
from common.logging_utils import get_logger

log = get_logger("ingest")


def _file_fingerprint(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def read_source(path: Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    if path.suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if path.suffix == ".json":
        return pd.read_json(path)
    raise ValueError(f"unsupported source type: {path.suffix}")


def discover(source_dir: Path, tables: tuple[str, ...]) -> dict[str, Path]:
    found = {}
    for t in tables:
        for ext in (".parquet", ".csv", ".json"):
            p = Path(source_dir) / f"{t}{ext}"
            if p.exists():
                found[t] = p
                break
    return found


def ingest(source_dir: Path | None = None, tables: tuple[str, ...] | None = None) -> dict:
    """Land raw sources into DuckDB `raw_*` tables and return a manifest."""
    source_dir = Path(source_dir or settings.RAW_DIR)
    tables = tables or ("users", "income", "transactions", "debts", "investments",
                        "financial_goals", "asset_prices")
    sources = discover(source_dir, tables)
    missing = [t for t in tables if t not in sources]
    if missing:
        raise FileNotFoundError(f"missing raw sources: {missing}. Run `make data` first.")

    manifest: dict = {
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_dir": str(source_dir),
        "tables": {},
    }
    con = duckdb_conn()
    for t, p in sources.items():
        t0 = time.time()
        df = read_source(p)
        con.execute(f"CREATE OR REPLACE TABLE raw_{t} AS SELECT * FROM df")
        manifest["tables"][t] = {
            "path": str(p),
            "rows": int(len(df)),
            "columns": list(map(str, df.columns)),
            "fingerprint": _file_fingerprint(p),
            "seconds": round(time.time() - t0, 3),
        }
        log.info("landed raw_%-16s %10s rows in %.2fs", t, f"{len(df):,}", time.time() - t0)
    con.close()

    (settings.PROCESSED_DIR / "ingestion_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(ingest(), indent=2))
