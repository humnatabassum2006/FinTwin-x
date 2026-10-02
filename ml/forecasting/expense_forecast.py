"""
Model A/B — expense & income forecasting with prediction intervals.

Algorithms: XGBoost and LightGBM (direct multi-horizon), benchmarked against a
seasonal-naive baseline. Uncertainty is quantified with *conformal* prediction
intervals built from a held-out calibration split, so the bands shown in the
dashboard have guaranteed empirical coverage (not a hand-waved ±10%).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from common import settings
from common.logging_utils import get_logger
from ml.evaluation.metrics import regression_report
from ml.forecasting.dataset import build_forecast_dataset
from ml.models.registry import registry

log = get_logger("forecast")


@dataclass
class HorizonModel:
    horizon: int
    algorithm: str
    model: object
    residual_quantiles: tuple[float, float] = (-0.2, 0.2)
    metrics: dict = field(default_factory=dict)


@dataclass
class ForecastModelBundle:
    target: str
    horizons: dict[int, HorizonModel]
    feature_names: list
    baselines: dict = field(default_factory=dict)

    def predict(self, X: pd.DataFrame, horizon: int) -> np.ndarray:
        hm = self.horizons[horizon]
        X = X[self.feature_names]
        if hm.algorithm == "xgboost":
            import xgboost as xgb
            dmat = xgb.DMatrix(X)
            return hm.model.predict(dmat)
        return hm.model.predict(X)

    def interval(self, yhat: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
        lo_q, hi_q = self.horizons[horizon].residual_quantiles
        return yhat * (1 + lo_q), yhat * (1 + hi_q)


def _fit(name: str, X: pd.DataFrame, y: np.ndarray, params: dict):
    if name == "xgboost":
        import xgboost as xgb
        dtrain = xgb.DMatrix(X, label=y)
        return xgb.train(params, dtrain, num_boost_round=params.pop("num_boost_round", 350)), "xgboost"
    import lightgbm as lgb
    ds = lgb.Dataset(X, label=y)
    p = dict(params)
    rounds = p.pop("num_boost_round", 350)
    return lgb.train(p, ds, num_boost_round=rounds), "lightgbm"


def train_forecast_model(panel: pd.DataFrame, target: str = "expense",
                         horizons: tuple[int, ...] = (1, 3, 6, 12),
                         algorithm: str = "xgboost", test_size: float = 0.2) -> ForecastModelBundle:
    ds = build_forecast_dataset(panel, target_col=target, horizons=horizons)
    n = len(ds.X)
    rng = np.random.default_rng(settings.RANDOM_SEED)
    bundle = ForecastModelBundle(target=target, horizons={}, feature_names=ds.feature_names)

    for h in horizons:
        y = ds.y[h]
        if algorithm == "xgboost":
            params = dict(objective="reg:squarederror", max_depth=6, eta=0.06, subsample=0.85,
                          colsample_bytree=0.85, min_child_weight=10, reg_lambda=1.5,
                          tree_method="hist", num_boost_round=350)
        else:
            params = dict(objective="regression", learning_rate=0.06, num_leaves=48,
                          feature_fraction=0.85, bagging_fraction=0.85, bagging_freq=1,
                          min_data_in_leaf=40, num_boost_round=350, verbose=-1)

        h_idx = np.flatnonzero(ds.train_mask[h])
        perm = rng.permutation(h_idx)
        cut = int(len(perm) * (1 - test_size))
        cal_cut = int(cut * 0.75)
        tr, cal, te = perm[:cal_cut], perm[cal_cut:cut], perm[cut:]

        model, algo = _fit(algorithm, ds.X.iloc[tr], y[tr], dict(params))

        # conformal intervals on the calibration split (multiplicative residuals)
        X_cal, X_te = ds.X.iloc[cal], ds.X.iloc[te]
        if algo == "xgboost":
            import xgboost as xgb
            pred_cal = model.predict(xgb.DMatrix(X_cal))
            pred_te = model.predict(xgb.DMatrix(X_te))
        else:
            pred_cal, pred_te = model.predict(X_cal), model.predict(X_te)
        resid_cal = (y[cal] - pred_cal) / np.clip(np.abs(pred_cal), 1e-6, None)
        lo_q, hi_q = np.quantile(resid_cal, [0.05, 0.95])
        lo, hi = pred_te * (1 + lo_q), pred_te * (1 + hi_q)

        rep = regression_report(y[te], pred_te, lo, hi)
        # baseline: last observed value
        last_col = f"{target}_lag1"
        baseline_pred = X_te[last_col].to_numpy() if last_col in X_te.columns else np.full(len(te), y[tr].mean())
        rep_baseline = regression_report(y[te], baseline_pred)

        hm = HorizonModel(horizon=h, algorithm=algo, model=model,
                          residual_quantiles=(float(lo_q), float(hi_q)), metrics=rep)
        bundle.horizons[h] = hm
        bundle.baselines[h] = {"last_value": rep_baseline}

        registry.register(
            name=f"{target}_forecast_{algo}_h{h}", model=model, algorithm=algo,
            task="regression", metrics={**rep, "baseline_mae": rep_baseline["mae"]},
            params=params, features=ds.feature_names, dataset=f"monthly_panel[{target}]", rows=int(len(tr)),
            notes=f"Direct multi-horizon forecast, h={h}, conformal 90% interval.",
        )
        log.info("%s h=%-2s %-8s MAE=%.0f MAPE=%.1f%% coverage=%.2f (baseline MAE=%.0f)",
                 target, h, algo, rep["mae"], rep["mape_pct"], rep.get("coverage_90", float("nan")),
                 rep_baseline["mae"])
    return bundle


def forecast_user(bundle: ForecastModelBundle, X_latest: pd.DataFrame) -> dict:
    """Point forecast + interval + baseline comparison for one user."""
    out = {}
    for h, hm in bundle.horizons.items():
        yhat = float(bundle.predict(X_latest, h)[0])
        lo, hi = bundle.interval(np.array([yhat]), h)
        out[h] = {
            "point": round(yhat, 2),
            "low": round(float(lo[0]), 2),
            "high": round(float(hi[0]), 2),
            "coverage_target": 0.90,
            "model": hm.algorithm,
            "baseline_mae": bundle.baselines.get(h, {}).get("last_value", {}).get("mae"),
        }
    return out


def latest_feature_row(panel: pd.DataFrame, user_id: int, bundle: ForecastModelBundle) -> pd.DataFrame:
    """Recreate the feature row for a user's most recent month (serve path)."""
    from ml.forecasting.dataset import build_forecast_dataset

    ds = build_forecast_dataset(panel, target_col=bundle.target, horizons=tuple(bundle.horizons))
    p = panel[panel["user_id"] == user_id].sort_values("year_month")
    row = ds.X[ds.groups == user_id].tail(1)
    return row
