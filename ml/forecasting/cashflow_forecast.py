"""
Model B — cash-flow forecast.

Cash flow is forecast *compositionally*: income and expenses are modelled
separately (they have very different dynamics) and the net cash flow is derived
as income − expense − EMI, with intervals propagated from both components.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from ml.forecasting.dataset import build_forecast_dataset
from ml.forecasting.expense_forecast import ForecastModelBundle, train_forecast_model

log = get_logger("forecast.cashflow")

STORE_PATH = settings.MODEL_DIR / "forecast_store.joblib"


class ForecastStore:
    """Trains once, serves many: holds the expense/income bundles + row index."""

    def __init__(self):
        self.expense: ForecastModelBundle | None = None
        self.income: ForecastModelBundle | None = None
        self.row_index_exp: dict = {}
        self.row_index_inc: dict = {}
        self.panel_months: list = []
        self.metrics: dict = {}

    # ---------------------------------------------------------------- fit
    def fit(self, panel: pd.DataFrame, algorithm: str = "xgboost",
            horizons: tuple[int, ...] = settings.FORECAST_HORIZONS) -> "ForecastStore":
        log.info("training forecasting models (%s) …", algorithm)
        self.expense = train_forecast_model(panel, target="expense", horizons=horizons, algorithm=algorithm)
        self.income = train_forecast_model(panel, target="income", horizons=horizons, algorithm=algorithm)

        ds_e = build_forecast_dataset(panel, target_col="expense", horizons=horizons)
        ds_i = build_forecast_dataset(panel, target_col="income", horizons=horizons)
        self.row_index_exp = pd.Series(np.arange(len(ds_e.X)), index=ds_e.groups).groupby(level=0).last().to_dict()
        self.row_index_inc = pd.Series(np.arange(len(ds_i.X)), index=ds_i.groups).groupby(level=0).last().to_dict()
        self._ds_e, self._ds_i = ds_e, ds_i
        self.panel_months = sorted(panel["year_month"].unique().tolist())

        self.metrics = {
            "expense": {str(h): {**hm.metrics, "baseline": self.expense.baselines[h]["last_value"]}
                        for h, hm in self.expense.horizons.items()},
            "income": {str(h): {**hm.metrics, "baseline": self.income.baselines[h]["last_value"]}
                       for h, hm in self.income.horizons.items()},
        }
        return self

    # ---------------------------------------------------------------- serve
    def _row(self, ds, row_index: dict, user_id: int) -> pd.DataFrame | None:
        i = row_index.get(int(user_id))
        return None if i is None else ds.X.iloc[[i]]

    def forecast_expenses(self, user_id: int) -> dict:
        row = self._row(self._ds_e, self.row_index_exp, user_id)
        if row is None:
            return {}
        out = {}
        for h, hm in self.expense.horizons.items():
            yhat = float(self.expense.predict(row, h)[0])
            lo, hi = self.expense.interval(np.array([yhat]), h)
            out[h] = {"point": round(yhat, 2), "low": round(float(lo[0]), 2),
                      "high": round(float(hi[0]), 2)}
        return out

    def forecast_cashflow(self, user_id: int, panel: pd.DataFrame | None = None) -> dict:
        row_e = self._row(self._ds_e, self.row_index_exp, user_id)
        row_i = self._row(self._ds_i, self.row_index_inc, user_id)
        if row_e is None or row_i is None:
            return {}
        emi = 0.0
        if panel is not None:
            u = panel[panel["user_id"] == user_id].sort_values("year_month").tail(1)
            emi = float(u["emi"].iloc[0]) if len(u) else 0.0
        out = {}
        for h in self.expense.horizons:
            inc = float(self.income.predict(row_i, h)[0])
            exp = float(self.expense.predict(row_e, h)[0])
            inc_lo, inc_hi = self.income.interval(np.array([inc]), h)
            exp_lo, exp_hi = self.expense.interval(np.array([exp]), h)
            net = inc - exp - emi
            out[h] = {
                "income": round(inc, 2), "expense": round(exp, 2), "emi": round(emi, 2),
                "net": round(net, 2),
                "net_low": round(float(inc_lo[0]) - float(exp_hi[0]) - emi, 2),
                "net_high": round(float(inc_hi[0]) - float(exp_lo[0]) - emi, 2),
                "expense_low": round(float(exp_lo[0]), 2), "expense_high": round(float(exp_hi[0]), 2),
                "income_low": round(float(inc_lo[0]), 2), "income_high": round(float(inc_hi[0]), 2),
            }
        return out

    # ---------------------------------------------------------------- persistence
    def save(self, path: Path | None = None) -> None:
        path = Path(path or STORE_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"expense": self.expense, "income": self.income,
                     "row_index_exp": self.row_index_exp, "row_index_inc": self.row_index_inc,
                     "metrics": self.metrics}, path)
        log.info("forecast store saved → %s", path)

    @classmethod
    def load(cls, panel: pd.DataFrame, path: Path | None = None) -> "ForecastStore":
        path = Path(path or STORE_PATH)
        store = cls()
        if not path.exists():
            store.fit(panel)
            store.save(path)
            return store
        blob = joblib.load(path)
        store.expense, store.income = blob["expense"], blob["income"]
        store.row_index_exp, store.row_index_inc = blob["row_index_exp"], blob["row_index_inc"]
        store.metrics = blob.get("metrics", {})
        ds_e = build_forecast_dataset(panel, target_col="expense", horizons=tuple(store.expense.horizons))
        ds_i = build_forecast_dataset(panel, target_col="income", horizons=tuple(store.income.horizons))
        store._ds_e, store._ds_i = ds_e, ds_i
        return store


_store: ForecastStore | None = None


def get_store(panel: pd.DataFrame | None = None) -> ForecastStore:
    """Process-wide singleton used by the API."""
    global _store
    if _store is None:
        panel = panel if panel is not None else pd.read_parquet(settings.PROCESSED_DIR / "monthly_panel.parquet")
        _store = ForecastStore.load(panel)
    return _store
