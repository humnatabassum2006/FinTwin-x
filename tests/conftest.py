"""Shared pytest fixtures.

Tests that need the full dataset skip automatically if it has not been built,
so `pytest` is safe to run right after a clone:

    make data-small && make pipeline && make train && pytest
"""
from __future__ import annotations

from pathlib import Path

import pytest

from common import settings


def _exists(name: str) -> bool:
    return (settings.PROCESSED_DIR / name).exists()


@pytest.fixture(scope="session")
def processed_ready() -> bool:
    return all(_exists(f) for f in ("monthly_panel.parquet", "financial_features.parquet",
                                    "training_dataset.parquet", "transactions.parquet"))


@pytest.fixture(scope="session")
def models_ready() -> bool:
    return all((settings.MODEL_DIR / f).exists()
               for f in ("risk_model.joblib", "forecast_store.joblib", "anomaly_model.joblib"))


@pytest.fixture(scope="session")
def panel(processed_ready):
    import pandas as pd
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    return pd.read_parquet(settings.PROCESSED_DIR / "monthly_panel.parquet")


@pytest.fixture(scope="session")
def features(processed_ready):
    import pandas as pd
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    return pd.read_parquet(settings.PROCESSED_DIR / "financial_features.parquet")


@pytest.fixture(scope="session")
def training(processed_ready):
    import pandas as pd
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    return pd.read_parquet(settings.PROCESSED_DIR / "training_dataset.parquet")


@pytest.fixture(scope="session")
def sample_tx(processed_ready):
    import pandas as pd
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    tx = pd.read_parquet(settings.PROCESSED_DIR / "transactions.parquet")
    ids = sorted(tx["user_id"].unique())[:200]
    return tx[tx["user_id"].isin(ids)]


@pytest.fixture(scope="session")
def users(processed_ready):
    import pandas as pd
    if not processed_ready:
        pytest.skip("run `make pipeline` first")
    return pd.read_parquet(settings.PROCESSED_DIR / "users.parquet")
