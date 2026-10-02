"""Meta, health, population and monitoring endpoints."""
from __future__ import annotations

import json
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Request

from agents.context import DataContext
from apps.api.deps import get_ctx, rate_limit
from common import settings
from common.db import table_stats
from ml.monitoring.drift import drift_report

router = APIRouter(tags=["meta"])

VERSION = "2.0.0"
_START = time.time()


@router.get("/health", summary="Service health + data/model availability")
def health(request: Request, ctx: DataContext = Depends(get_ctx)):
    proc_dir = settings.PROCESSED_DIR
    models = {
        "risk_model": (settings.MODEL_DIR / "risk_model.joblib").exists(),
        "forecast_store": (settings.MODEL_DIR / "forecast_store.joblib").exists(),
        "anomaly_model": (settings.MODEL_DIR / "anomaly_model.joblib").exists(),
    }
    tables = table_stats() if (proc_dir / "financial_features.parquet").exists() else {}
    return {
        "status": "ok" if ctx.is_ready() else "data_missing",
        "version": VERSION,
        "as_of": str(settings.AS_OF),
        "uptime_seconds": round(time.time() - _START, 1),
        "tables": tables,
        "models": models,
        "llm_enabled": bool(settings.LLM_API_KEY),
        "auth_required": False,
        "environment": settings.ENVIRONMENT,
    }


@router.get("/metrics", summary="Model evaluation metrics (from the last training run)")
def metrics():
    p = settings.PROCESSED_DIR / "model_metrics.json"
    if not p.exists():
        raise HTTPException(404, "no metrics yet — run `make train`")
    return json.loads(p.read_text(encoding="utf-8"))


@router.get("/population", summary="Synthetic population summary")
def population(ctx: DataContext = Depends(get_ctx)):
    f = ctx.features
    return {
        "n_users": int(len(f)),
        "as_of": str(settings.AS_OF),
        "persona_mix": f["persona"].value_counts().to_dict(),
        "segment_mix": f["segment_label"].value_counts().to_dict(),
        "city_mix": f["city"].value_counts().to_dict(),
        "median_income": float(f["monthly_income"].median()),
        "median_expense": float(f["monthly_expense"].median()),
        "mean_health_score": float(f["dna_financial_health"].mean()),
        "mean_risk_score": float(f["dna_financial_risk"].mean()),
        "pct_with_debt": float((f["has_debt"] > 0).mean()),
    }


@router.get("/users", summary="Browse synthetic users (for demos)")
def list_users(limit: int = 25, segment: str | None = None, ctx: DataContext = Depends(get_ctx)):
    f = ctx.features
    if segment:
        f = f[f["segment_label"] == segment]
    cols = ["user_id", "persona", "segment_label", "age", "city", "monthly_income",
            "monthly_expense", "savings_rate", "emergency_fund_months", "dna_financial_health",
            "dna_financial_risk"]
    return {"users": json.loads(f[cols].head(limit).to_json(orient="records"))}


@router.get("/monitoring/drift", summary="Data drift vs the training distribution (PSI)")
def drift(limit_features: int = 25, ctx: DataContext = Depends(get_ctx)):
    try:
        return drift_report()
    except FileNotFoundError as exc:
        raise HTTPException(404, str(exc)) from exc


@router.get("/audit", summary="Recent audit-log entries")
def audit_tail(n: int = 50):
    from common.logging_utils import audit
    return {"entries": audit.tail(n)}


@router.get("/tools", summary="Agent tool manifest")
def tools_manifest():
    from agents.tools import tool_manifest
    return {"tools": tool_manifest()}
