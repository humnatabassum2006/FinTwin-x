"""
Model D — anomaly detection.

Three complementary detectors, because no single one is trustworthy alone:

  1. Isolation Forest on transaction-level behavioural features
  2. Robust statistics (median + MAD z-score) per (user, category)
  3. Burst detector (several unusual transactions in a short window)

The fused score is evaluated against the injected ground truth using
precision@K / recall@K — the metric that actually matters for an alerting
system, since an analyst only ever looks at the top K alerts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import QuantileTransformer

from common import settings
from common.logging_utils import get_logger
from ml.evaluation.metrics import anomaly_report
from ml.models.registry import registry

log = get_logger("anomaly")

FEATURES = [
    "amount_log", "ratio_to_own_median", "ratio_to_month_total", "mad_z",
    "burst_count", "merchant_rarity", "hour", "is_weekend", "days_since_prev",
    "category_share_of_month", "tx_count_month",
]


@dataclass
class AnomalyModel:
    forest: IsolationForest
    scaler: QuantileTransformer
    features: list = field(default_factory=lambda: list(FEATURES))
    metrics: dict = field(default_factory=dict)
    contamination: float = 0.02


# --------------------------------------------------------------------------------------
# feature engineering (vectorised over the whole ledger)
# --------------------------------------------------------------------------------------
def build_transaction_features(tx: pd.DataFrame) -> pd.DataFrame:
    df = tx.copy()
    df["ym"] = df["timestamp"].to_numpy().astype("datetime64[M]")
    df["date"] = df["timestamp"].to_numpy().astype("datetime64[D]")
    df["amount_log"] = np.log1p(df["amount"].astype("float64"))
    cat = df["category"].astype(str)

    grp_uc = df.groupby(["user_id", cat.name if False else "category"], observed=True)["amount"]
    median = grp_uc.transform("median").astype("float64")
    mad = (df["amount"] - median).abs().groupby(
        [df["user_id"], df["category"].astype(str)], observed=False).transform("median").astype("float64")
    df["ratio_to_own_median"] = df["amount"] / (median + 1.0)
    df["mad_z"] = 0.6745 * (df["amount"] - median) / (mad + 1.0)

    month_total = df.groupby(["user_id", "ym"], observed=True)["amount"].transform("sum")
    df["ratio_to_month_total"] = df["amount"] / (month_total + 1.0)
    df["tx_count_month"] = df.groupby(["user_id", "ym"], observed=True)["amount"].transform("size")
    df["category_share_of_month"] = df.groupby(["user_id", "ym", "category"], observed=True)["amount"].transform("sum") / (month_total + 1.0)

    df["burst_count"] = df.groupby(["user_id", "date", "category"], observed=True)["amount"].transform("size")
    df["merchant_rarity"] = df.groupby(["user_id", "merchant"], observed=True)["amount"].transform("size")
    df["hour"] = df["timestamp"].dt.hour.astype("int16")
    df["dow"] = df["timestamp"].dt.dayofweek.astype("int16")
    df["is_weekend"] = df["dow"].isin([5, 6]).astype("int8")
    prev = df.sort_values(["user_id", "category", "timestamp"]).groupby(
        ["user_id", "category"], observed=True)["timestamp"].shift(1)
    df["days_since_prev"] = ((df["timestamp"] - prev).dt.total_seconds() / 86400).fillna(30).clip(0, 400)
    return df


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------
def train_anomaly_model(tx: pd.DataFrame, sample_size: int = 1_200_000,
                        contamination: float = 0.02) -> AnomalyModel:
    """Fit on a USER-level sample of the ledger.

    Sampling whole users (instead of random rows) keeps the per-user history
    intact, which the ratio / MAD features depend on — and keeps memory flat on
    multi-million-row ledgers.
    """
    rng = np.random.default_rng(settings.RANDOM_SEED)
    n_total = len(tx)
    if n_total > sample_size:
        users = np.unique(tx["user_id"].to_numpy())
        rng.shuffle(users)
        keep_users, total = [], 0
        for u in users:
            k = int((tx["user_id"].to_numpy() == u).sum())
            if total + k > sample_size * 1.6:
                break
            keep_users.append(u)
            total += k
            if total >= sample_size:
                break
        tx = tx[tx["user_id"].isin(keep_users)].reset_index(drop=True)
    feats = build_transaction_features(tx)
    n = len(feats)
    idx = np.arange(n)

    X = feats[FEATURES].astype("float64")
    scaler = QuantileTransformer(output_distribution="uniform", n_quantiles=min(2000, len(idx)),
                                 random_state=settings.RANDOM_SEED)
    Xs = scaler.fit(X.iloc[idx])
    forest = IsolationForest(n_estimators=200, contamination=contamination,
                             random_state=settings.RANDOM_SEED, n_jobs=-1)
    forest.fit(scaler.transform(X.iloc[idx]))

    model = AnomalyModel(forest=forest, scaler=scaler, contamination=contamination)
    feats["if_score"] = _score_batch(model, X)
    feats = _fuse(feats)
    score = feats["anomaly_score"].to_numpy()
    if "is_anomaly" in feats.columns:
        y = feats["is_anomaly"].astype(int).to_numpy()
        ks = [k for k in (100, 500, 1000, 5000) if k <= n]
        model.metrics = anomaly_report(y, score, ks=ks)
        model.metrics["n_transactions"] = int(n)
        model.metrics["n_true_anomalies"] = int(y.sum())
        log.info("anomaly model: %s", {k: model.metrics[k] for k in list(model.metrics)[:4]})
        registry.register(
            name="anomaly_isolation_forest", model=model, algorithm="isolation_forest",
            task="anomaly_detection", metrics=model.metrics,
            params={"n_estimators": 200, "contamination": contamination,
                    "sample_size": int(min(sample_size, n))},
            features=FEATURES, dataset="transactions", rows=int(min(sample_size, n)),
            notes="Fused IF + robust-MAD + burst scoring, evaluated against injected ground truth.",
        )
    return model


def _score_batch(model: AnomalyModel, X: pd.DataFrame, batch: int = 500_000) -> np.ndarray:
    out = np.empty(len(X), dtype="float64")
    for i in range(0, len(X), batch):
        sl = slice(i, min(i + batch, len(X)))
        Xs = model.scaler.transform(X.iloc[sl])
        out[sl] = -model.forest.score_samples(Xs)          # higher = more anomalous
    return out


def _fuse(feats: pd.DataFrame) -> pd.DataFrame:
    """Ensemble of detectors, combined on the rank scale (scale-free, robust)."""
    r = lambda s_: pd.Series(np.asarray(s_, dtype="float64")).rank(pct=True).to_numpy()
    r_mad = r(feats["mad_z"].abs())
    r_month = r(feats["ratio_to_month_total"])
    r_if = r(feats["if_score"])
    r_burst = r(np.log1p(feats["burst_count"]))
    r_rare = r(1.0 / (feats["merchant_rarity"] + 1.0))
    feats["anomaly_score"] = (0.34 * r_mad + 0.22 * r_month + 0.22 * r_if +
                              0.12 * r_burst + 0.10 * r_rare)
    return feats


def score_transactions(model: AnomalyModel, tx: pd.DataFrame) -> pd.DataFrame:
    feats = build_transaction_features(tx)
    X = feats[FEATURES].astype("float64")
    feats["if_score"] = _score_batch(model, X)
    return _fuse(feats)


# --------------------------------------------------------------------------------------
# serving
# --------------------------------------------------------------------------------------
def user_anomalies(model: AnomalyModel, tx: pd.DataFrame, user_id: int, top_k: int = 10) -> list[dict]:
    u = tx[tx["user_id"] == user_id]
    if u.empty:
        return []
    scored = score_transactions(model, u)
    scored = scored.sort_values("anomaly_score", ascending=False).head(top_k)
    median_by_cat = u.groupby("category", observed=True)["amount"].median()
    out = []
    for _, r in scored.iterrows():
        typical = float(median_by_cat.get(r["category"], 0.0))
        out.append({
            "transaction_id": int(r["transaction_id"]),
            "timestamp": r["timestamp"].isoformat(),
            "amount": round(float(r["amount"]), 2),
            "category": str(r["category"]),
            "merchant": str(r["merchant"]),
            "score": round(float(r["anomaly_score"]), 4),
            "typical_amount": round(typical, 2),
            "excess": round(float(r["amount"]) - typical, 2),
            "severity": "high" if r["anomaly_score"] > 0.97 else ("medium" if r["anomaly_score"] > 0.90 else "low"),
            "detector_votes": {
                "isolation_forest": round(float(r["if_score"]), 4),
                "robust_z": round(float(r["mad_z"]), 2),
                "burst": int(r["burst_count"]),
            },
        })
    return out


def category_month_alerts(tx: pd.DataFrame, user_id: int, lookback: int = 6) -> list[dict]:
    """Simple, explainable alert: this month's category spend vs the trailing norm."""
    u = tx[tx["user_id"] == user_id].copy()
    if u.empty:
        return []
    u["ym"] = u["timestamp"].to_numpy().astype("datetime64[M]")
    piv = u.pivot_table(index="ym", columns="category", values="amount", aggfunc="sum", observed=True).fillna(0)
    if len(piv) < 3:
        return []
    latest = piv.iloc[-1]
    baseline = piv.iloc[-(lookback + 1):-1].median()
    alerts = []
    for cat in piv.columns:
        base = float(baseline.get(cat, 0))
        cur = float(latest.get(cat, 0))
        if base <= 0 or cur <= 0:
            continue
        ratio = cur / base
        if ratio >= 1.8 and (cur - base) > 5000:
            alerts.append({
                "category": str(cat), "current": round(cur, 2), "baseline": round(base, 2),
                "ratio": round(ratio, 2), "excess": round(cur - base, 2),
                "message": f"Unusual spending detected in {cat}: Rs {cur:,.0f} vs a typical Rs {base:,.0f} ({ratio:.1f}x).",
            })
    return sorted(alerts, key=lambda a: -a["ratio"])


def save_anomaly_model(model: AnomalyModel, path: Path | None = None) -> Path:
    path = Path(path or settings.MODEL_DIR / "anomaly_model.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_anomaly_model(path: Path | None = None):
    path = Path(path or settings.MODEL_DIR / "anomaly_model.joblib")
    return joblib.load(path) if path.exists() else None
