"""Simulation endpoints: Monte-Carlo, scenario comparison, optimisation, goals."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from agents import tools
from agents.context import DataContext
from apps.api.deps import get_ctx
from apps.api.schemas.models import (CompareRequest, OptimiseRequest, SimulateRequest)
from data_generator.scenarios import list_templates
from simulation.scenario_engine import preset_specs

router = APIRouter(tags=["simulation"])


def _safe(fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except KeyError as exc:
        raise HTTPException(404, {"error": f"user not found: {exc}"}) from exc
    except Exception as exc:                                   # noqa: BLE001
        raise HTTPException(500, {"error": str(exc)}) from exc


@router.get("/scenarios/presets", summary="Pre-built what-if scenarios")
def presets(user_id: int = 1, ctx: DataContext = Depends(get_ctx)):
    row = ctx.row(user_id)
    specs = preset_specs(row)
    return {"presets": [s.model_dump() for s in specs], "templates": list_templates()}


@router.post("/simulate", summary="Run a Monte-Carlo simulation for one scenario")
def simulate(req: SimulateRequest, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.run_monte_carlo, req.user_id,
                 req.scenario.model_dump() if req.scenario else None,
                 req.n_paths, req.horizon_months, ctx)


@router.post("/simulate/compare", summary="Compare scenarios side by side")
def compare(req: CompareRequest, ctx: DataContext = Depends(get_ctx)):
    scen = [s.model_dump() for s in req.scenarios]
    return _safe(tools.compare_scenarios, req.user_id, scen, req.n_paths,
                 req.horizon_months, ctx)


@router.post("/simulate/optimise", summary="Find the safest way to reach a target")
def optimise(req: OptimiseRequest, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.optimise_decision, req.user_id, req.scenario.model_dump(),
                 req.target_metric, req.target_value, req.n_paths, req.horizon_months, ctx)


@router.get("/users/{user_id}/goal", summary="Goal probability and required contribution")
def goal(user_id: int, n_paths: int = 4000, ctx: DataContext = Depends(get_ctx)):
    return _safe(tools.goal_outlook, user_id, n_paths, ctx)
