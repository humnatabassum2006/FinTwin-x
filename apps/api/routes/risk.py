"""Risk, explainability and stress-testing endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agents import tools
from agents.context import DataContext
from apps.api.deps import get_ctx

router = APIRouter(prefix="/users/{user_id}", tags=["risk"])


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as exc:
        raise HTTPException(404, {"error": f"user not found: {exc}"}) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(500, {"error": str(exc)}) from exc


@router.get("/risk", summary="FinTwin Risk Score with SHAP attribution")
def risk(user_id: int, explain: bool = True, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.calculate_risk, user_id, explain, ctx)


@router.get("/risk/explain", summary="Why is this the score? (risk points per driver)")
def explain(user_id: int, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.explain_risk, user_id, ctx)


@router.get("/risk/counterfactuals", summary="What can I change? (counterfactual actions)")
def counterfactuals(user_id: int, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.counterfactual_options, user_id, ctx)


@router.get("/stress-test", summary="Battery of adverse scenarios")
def stress(user_id: int, horizon_months: int = 36, n_paths: int = 3000,
           ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.stress_test, user_id, horizon_months, n_paths, ctx)
