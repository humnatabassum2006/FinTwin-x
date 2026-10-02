"""Data generator + pipeline tests: integrity, determinism, no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from common import settings
from data_generator import generate as gen
from data_generator.personas import PERSONAS, PERSONA_KEYS
from pipelines.features.build_features import DNA_DIMENSIONS
from pipelines.validation.validate import ExpectationResult, validate_table


def test_persona_weights_sum_to_one():
    assert abs(sum(PERSONAS[k].weight for k in PERSONA_KEYS) - 1.0) < 1e-9


def test_generator_is_deterministic():
    """Same seed → identical population (reproducibility is non-negotiable)."""
    a = gen.build_population(40, 12, seed=99)["users"][["user_id", "persona", "monthly_income_base"]]
    b = gen.build_population(40, 12, seed=99)["users"][["user_id", "persona", "monthly_income_base"]]
    pd.testing.assert_frame_equal(a, b)


def test_generator_respects_personas():
    pop = gen.build_population(300, 12, seed=7)
    users = pop["users"]
    assert set(users["persona"]).issubset(set(PERSONA_KEYS))
    assert users["user_id"].is_unique
    assert (users["monthly_income_base"] > 0).all()
    # income ordering: investors should out-earn the fragile archetype on average
    inv = users.loc[users.persona == "aggressive_investor", "monthly_income_base"].median()
    fra = users.loc[users.persona == "financially_fragile", "monthly_income_base"].median()
    assert inv > fra


def test_transaction_ledger_integrity():
    pop = gen.build_population(120, 12, seed=3)
    tx = pop["transactions"]
    assert len(tx) > 0
    assert (tx["amount"] > 0).all()
    assert set(tx["category"].astype(str)) <= set(settings.EXPENSE_CATEGORIES)
    assert tx["timestamp"].max() <= pd.Timestamp(settings.AS_OF)
    assert tx["transaction_id"].is_unique
    # ground-truth anomalies exist and are typed
    anom = tx[tx["is_anomaly"] == 1]
    if len(anom):
        assert set(anom["anomaly_type"]) <= {"card_fraud_burst", "medical_emergency",
                                             "big_ticket_purchase", "travel_spike"}


def test_validation_catches_bad_rows():
    good = pd.DataFrame({"user_id": [1, 2], "amount": [100.0, 200.0],
                         "timestamp": pd.to_datetime(["2026-01-01", "2026-02-01"]),
                         "category": ["Food", "Housing"], "merchant": ["a", "b"],
                         "payment_method": ["cash", "cash"], "transaction_type": ["debit", "debit"],
                         "is_anomaly": [0, 0], "anomaly_type": ["none", "none"],
                         "transaction_id": [1, 2]})
    results, _ = validate_table(good, "transactions", {})
    assert all(r.success for r in results if r.severity == "error")

    bad = good.copy()
    bad.loc[0, "amount"] = -50.0           # negative amount
    bad.loc[1, "category"] = "CryptoNFT"   # unknown category
    results2, clean = validate_table(bad, "transactions", {})
    failures = [r for r in results2 if not r.success and r.severity == "error"]
    assert failures, "validation should reject negative amounts and unknown categories"


def test_validation_quarantines_duplicates():
    df = pd.DataFrame({"user_id": [1, 1, 2], "amount": [10.0, 10.0, 20.0],
                       "timestamp": pd.to_datetime(["2026-01-01"] * 3),
                       "category": ["Food"] * 3, "merchant": ["a"] * 3,
                       "payment_method": ["cash"] * 3, "transaction_type": ["debit"] * 3,
                       "is_anomaly": [0] * 3, "anomaly_type": ["none"] * 3,
                       "transaction_id": [1, 1, 2]})
    _, clean = validate_table(df, "transactions", {})
    assert len(clean) < len(df)


def test_panel_consistency(panel, features):
    assert len(panel) == panel[["user_id", "year_month"]].drop_duplicates().shape[0]
    merged = panel.groupby("user_id")["net_cash_flow"].mean()
    assert np.isfinite(merged).all()
    assert len(features) == features["user_id"].nunique()
    assert (features["monthly_income"] > 0).all()


def test_financial_dna_bounds(features):
    for d in DNA_DIMENSIONS:
        col = f"dna_{d}"
        assert col in features.columns, col
        assert features[col].between(0, 100).all(), col


def test_no_target_leakage(training):
    """No column may encode the label itself."""
    assert "stress_label" in training.columns
    suspicious = [c for c in training.columns if "stress" in c and c != "stress_label"]
    assert not suspicious, f"leakage columns: {suspicious}"
    assert training["stress_label"].isin([0, 1]).all()
    assert 0.02 < training["stress_label"].mean() < 0.98


def test_features_are_finite(features):
    num = features.select_dtypes(include=[np.number])
    assert np.isfinite(num.to_numpy()).all(), "NaN/inf in the feature store"
