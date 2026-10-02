"""
Pydantic contracts for every domain table.

These are the single source of truth for validation: the ingestion layer
validates raw rows against them, and the API uses them for request/response
serialisation (so the same rules hold at the edge and in the pipeline).
"""
from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from common.config import settings

CATEGORIES = settings.EXPENSE_CATEGORIES


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, validate_assignment=True)


class UserRecord(StrictModel):
    user_id: int = Field(gt=0)
    persona: str
    age: int = Field(ge=18, le=90)
    household_size: int = Field(ge=1, le=12)
    city: str
    city_cost_index: float = Field(ge=0.5, le=2.0)
    income_type: Literal["salaried", "business", "freelance", "rental", "mixed"]
    employment_type: str
    financial_goal: str
    risk_preference: Literal["conservative", "balanced", "aggressive"]
    monthly_income_base: float = Field(gt=0)
    opening_cash_balance: float = Field(ge=0)
    monthly_expense_estimate: float = Field(ge=0)
    created_at: datetime


class IncomeRecord(StrictModel):
    income_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    date: datetime
    amount: float = Field(gt=0)
    source: str
    income_type: str
    recurring: bool


class TransactionRecord(StrictModel):
    transaction_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    timestamp: datetime
    amount: float = Field(gt=0, le=50_000_000)
    category: str
    merchant: str
    payment_method: str
    transaction_type: Literal["debit", "credit"]
    is_anomaly: int = Field(ge=0, le=1)
    anomaly_type: str = "none"

    @field_validator("category")
    @classmethod
    def _cat(cls, v: str) -> str:
        if v not in CATEGORIES:
            raise ValueError(f"unknown category {v!r}")
        return v


class DebtRecord(StrictModel):
    loan_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    loan_type: str
    principal: float = Field(gt=0)
    interest_rate: float = Field(ge=0, le=1)
    monthly_payment: float = Field(ge=0)
    remaining_balance: float = Field(ge=0)
    start_date: date
    end_date: date
    term_months: int = Field(gt=0)

    @field_validator("end_date")
    @classmethod
    def _end_after_start(cls, v: date, info) -> date:
        start = info.data.get("start_date")
        if start and v <= start:
            raise ValueError("end_date must be after start_date")
        return v


class InvestmentRecord(StrictModel):
    investment_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    asset: str
    quantity: float = Field(gt=0)
    purchase_price: float = Field(gt=0)
    current_price: float = Field(gt=0)
    purchase_date: date
    as_of_date: date
    liquidity_score: float = Field(ge=0, le=1)
    expected_return: float
    volatility: float = Field(ge=0)


class GoalRecord(StrictModel):
    goal_id: int = Field(gt=0)
    user_id: int = Field(gt=0)
    goal_name: str
    target_amount: float = Field(gt=0)
    current_amount: float = Field(ge=0)
    target_date: date
    monthly_contribution: float = Field(ge=0)
    priority: int = Field(ge=1, le=5)
    months_elapsed: int = Field(ge=0)
    status: str


class AssetPriceRecord(StrictModel):
    date: date
    asset: str
    price_index: float = Field(gt=0)


SCHEMAS = {
    "users": UserRecord,
    "income": IncomeRecord,
    "transactions": TransactionRecord,
    "debts": DebtRecord,
    "investments": InvestmentRecord,
    "financial_goals": GoalRecord,
    "asset_prices": AssetPriceRecord,
}

PII_COLUMNS = ("merchant",)   # merchant strings are the only quasi-identifier we keep
