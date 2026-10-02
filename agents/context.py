"""
Shared in-process context: processed data + trained models.

Both the REST API and the agent layer read from the same cached objects, so an
agent's answer and an endpoint's response can never disagree.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from common import settings
from common.logging_utils import get_logger

log = get_logger("context")


class DataContext:
    def __init__(self):
        self._features: pd.DataFrame | None = None
        self._panel: pd.DataFrame | None = None
        self._tx: pd.DataFrame | None = None
        self._users: pd.DataFrame | None = None
        self._goals: pd.DataFrame | None = None
        self._risk = None
        self._forecast = None
        self._anomaly = None
        self._background = None

    # ------------------------------------------------------------------ data
    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            self._features = pd.read_parquet(settings.PROCESSED_DIR / "financial_features.parquet")
        return self._features

    @property
    def panel(self) -> pd.DataFrame:
        if self._panel is None:
            self._panel = pd.read_parquet(settings.PROCESSED_DIR / "monthly_panel.parquet")
        return self._panel

    @property
    def transactions(self) -> pd.DataFrame:
        if self._tx is None:
            self._tx = pd.read_parquet(settings.PROCESSED_DIR / "transactions.parquet")
        return self._tx

    @property
    def users(self) -> pd.DataFrame:
        if self._users is None:
            self._users = pd.read_parquet(settings.PROCESSED_DIR / "users.parquet")
        return self._users

    @property
    def goals(self) -> pd.DataFrame:
        if self._goals is None:
            self._goals = pd.read_parquet(settings.PROCESSED_DIR / "financial_goals.parquet")
        return self._goals

    # ------------------------------------------------------------------ models
    @property
    def risk_model(self):
        if self._risk is None:
            from ml.risk.risk_model import load_risk_model
            self._risk = load_risk_model()
        return self._risk

    @property
    def forecast_store(self):
        if self._forecast is None:
            from ml.forecasting.cashflow_forecast import get_store
            self._forecast = get_store(self.panel)
        return self._forecast

    @property
    def anomaly_model(self):
        if self._anomaly is None:
            from ml.anomaly_detection.anomaly import load_anomaly_model
            self._anomaly = load_anomaly_model()
        return self._anomaly

    @property
    def background(self):
        if self._background is None:
            p = settings.MODEL_DIR / "risk_background.parquet"
            self._background = pd.read_parquet(p) if p.exists() else None
        return self._background

    # ------------------------------------------------------------------ helpers
    def row(self, user_id: int) -> pd.Series:
        df = self.features
        m = df[df["user_id"] == int(user_id)]
        if m.empty:
            raise KeyError(f"user_id {user_id} not found")
        return m.iloc[0]

    def user_ids(self, n: int = 50) -> list[int]:
        return self.features["user_id"].head(n).astype(int).tolist()

    def panel_for(self, user_id: int) -> pd.DataFrame:
        return self.panel[self.panel["user_id"] == int(user_id)].sort_values("year_month")

    def tx_for(self, user_id: int) -> pd.DataFrame:
        return self.transactions[self.transactions["user_id"] == int(user_id)]

    def is_ready(self) -> bool:
        return all((settings.PROCESSED_DIR / f"{t}.parquet").exists()
                   for t in ("financial_features", "monthly_panel", "transactions"))


@lru_cache(maxsize=1)
def get_context() -> DataContext:
    return DataContext()
