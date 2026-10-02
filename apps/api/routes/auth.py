"""Authentication, authorisation and audit endpoints."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from apps.api.deps import current_user, rate_limit, require_user
from apps.api.schemas.models import LoginRequest, RegisterRequest, TokenResponse
from apps.api.security import (authenticate, create_token, register_user,
                               seed_demo_user)
from common import settings
from common.logging_utils import audit

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register", response_model=TokenResponse, dependencies=[Depends(rate_limit)])
def register(req: RegisterRequest):
    try:
        user = register_user(req.username, req.email, req.password, req.fintwin_user_id)
    except ValueError as exc:
        raise HTTPException(400, {"error": str(exc)}) from exc
    token = create_token(user.username, user.role, user.fintwin_user_id)
    return TokenResponse(access_token=token, expires_in=settings.JWT_EXPIRE_MINUTES * 60,
                         user={"username": user.username, "role": user.role,
                               "fintwin_user_id": user.fintwin_user_id})


@router.post("/login", response_model=TokenResponse, dependencies=[Depends(rate_limit)])
def login(req: LoginRequest):
    if settings.ENABLE_DEMO_USER:
        seed_demo_user()
    user = authenticate(req.username, req.password)
    if user is None:
        raise HTTPException(401, {"error": "invalid credentials"})
    token = create_token(user.username, user.role, user.fintwin_user_id)
    return TokenResponse(access_token=token, expires_in=settings.JWT_EXPIRE_MINUTES * 60,
                         user={"username": user.username, "role": user.role,
                               "fintwin_user_id": user.fintwin_user_id})


@router.get("/me")
def me(claims=Depends(require_user)):
    return {"user": claims}


@router.get("/demo-credentials")
def demo_credentials():
    if not settings.ENABLE_DEMO_USER:
        raise HTTPException(404, {"error": "demo account is disabled"})
    seed_demo_user()
    return {"username": "demo", "password": "fintwin-demo-2026",
            "note": "Demo account for the sandbox only — synthetic data, no real credentials."}
