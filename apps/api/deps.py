"""Shared FastAPI dependencies: rate limiting, optional auth, data context."""
from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, Header, HTTPException, Request

from agents.context import DataContext, get_context
from apps.api.security import decode_token
from common import settings
from common.logging_utils import audit, get_logger

log = get_logger("api.deps")


class RateLimiter:
    """In-memory sliding-window limiter (per IP). Swap for Redis in production."""

    def __init__(self, per_minute: int = settings.RATE_LIMIT_PER_MINUTE):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> tuple[bool, int]:
        now = time.time()
        q = self.hits[key]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= self.per_minute:
            return False, len(q)
        q.append(now)
        return True, len(q)


limiter = RateLimiter()


def rate_limit(request: Request) -> None:
    key = request.client.host if request.client else "anonymous"
    ok, count = limiter.check(key)
    if not ok:
        audit.write(actor=key, action="rate_limit", resource=request.url.path, status="blocked")
        raise HTTPException(status_code=429, detail={
            "error": "rate_limited",
            "limit_per_minute": limiter.per_minute,
            "observed": count,
        })


def get_ctx() -> DataContext:
    return get_context()


def current_user(authorization: str | None = Header(default=None)) -> dict | None:
    """Optional auth: returns the decoded claims or None (demo mode)."""
    if not authorization:
        return None
    token = authorization.replace("Bearer ", "").strip()
    claims = decode_token(token)
    if claims is None:
        raise HTTPException(status_code=401, detail="invalid or expired token")
    return claims


def require_user(claims: dict | None = Depends(current_user)) -> dict:
    if claims is None:
        raise HTTPException(status_code=401, detail="authentication required")
    return claims


def require_role(role: str):
    def _dep(claims: dict = Depends(require_user)) -> dict:
        if claims.get("role") != role and claims.get("role") != "admin":
            raise HTTPException(status_code=403, detail="insufficient permissions")
        return claims
    return _dep
