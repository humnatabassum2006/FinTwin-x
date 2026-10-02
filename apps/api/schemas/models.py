"""Pydantic request/response contracts for the FinTwin-X API."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from simulation.scenario_engine import ScenarioSpec


class LoginRequest(BaseModel):
    username: str = Field(min_length=3, max_length=64)
    password: str = Field(min_length=8, max_length=128)


class RegisterRequest(LoginRequest):
    email: str
    fintwin_user_id: int = 1


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: dict


class ScenarioRequest(ScenarioSpec):
    pass


class CompareRequest(BaseModel):
    user_id: int
    scenarios: list[ScenarioSpec] = Field(default_factory=list)
    n_paths: int = 4000
    horizon_months: int = 36


class SimulateRequest(BaseModel):
    user_id: int
    scenario: ScenarioSpec | None = None
    n_paths: int = 5000
    horizon_months: int = 36


class OptimiseRequest(BaseModel):
    user_id: int
    scenario: ScenarioSpec
    target_metric: str = "goal_probability"
    target_value: float = 0.75
    n_paths: int = 2500
    horizon_months: int = 36


class AgentRequest(BaseModel):
    user_id: int = Field(default=1, ge=1)
    question: str = Field(min_length=1, max_length=1000)

    @field_validator("question")
    @classmethod
    def normalise_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("question must not be blank")
        return value


class RagRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    k: int = 4


class RagAnswerRequest(RagRequest):
    pass


class HealthResponse(BaseModel):
    status: str
    version: str
    as_of: str
    tables: dict
    models: dict
    llm_enabled: bool
    auth_required: bool
