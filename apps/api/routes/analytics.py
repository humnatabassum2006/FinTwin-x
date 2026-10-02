"""Analytics endpoints: overview, cash flow, spending, DNA, forecast, anomalies."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agents import tools
from agents.context import DataContext
from apps.api.deps import get_ctx
from common.config import settings

router = APIRouter(prefix="/users/{user_id}", tags=["analytics"])


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as exc:
        raise HTTPException(404, {"error": f"user not found: {exc}"}) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(500, {"error": str(exc)}) from exc


@router.get("/overview", summary="Executive overview for one user")
def overview(user_id: int, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.get_user_snapshot, user_id, ctx)


@router.get("/dna", summary="Financial DNA profile (9 dimensions)")
def dna(user_id: int, ctx: DataContext = Depends(get_ctx)):
    row = _safe(ctx.row, user_id)
    dims = ["liquidity", "savings_discipline", "spending_stability", "debt_resilience",
            "income_stability", "investment_exposure", "goal_discipline",
            "emergency_resilience", "financial_risk"]
    return {
        "user_id": int(user_id),
        "dimensions": [{"key": d, "label": d.replace("_", " ").title(),
                        "score": round(float(row[f"dna_{d}"]), 1)} for d in dims],
        "health_score": round(float(row["dna_financial_health"]), 1),
        "risk_score_rule": round(float(row["dna_financial_risk"]), 1),
        "segment": str(row.get("segment_label", "")),
        "persona": str(row.get("persona", "")),
        "population_medians": {d: round(float(ctx.features[f"dna_{d}"].median()), 1) for d in dims},
    }


@router.get("/cashflow", summary="Historical cash-flow analysis")
def cashflow(user_id: int, months: int = 12, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.calculate_cashflow, user_id, months, ctx)


@router.get("/spending", summary="Spending intelligence: categories, merchants, behaviour")
def spending(user_id: int, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.spending_intelligence, user_id, ctx)


@router.get("/forecast", summary="Expense & cash-flow forecast with intervals")
def forecast(user_id: int, ctx: DataContext = Depends(get_ctx)):
    return {"expenses": _safe(tools.forecast_expenses, user_id, (1, 3, 6, 12), ctx),
            "cashflow": _safe(tools.forecast_cashflow, user_id, ctx)}


@router.get("/anomalies", summary="Anomaly detection results")
def anomalies(user_id: int, top_k: int = 8, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.detect_anomalies, user_id, top_k, ctx)


@router.get("/goals", summary="Goals for the user")
def goals(user_id: int, ctx: DataContext = Depends(get_ctx)):
    g = ctx.goals
    g = g[g["user_id"] == int(user_id)]
    return {"goals": g.to_dict(orient="records")}
