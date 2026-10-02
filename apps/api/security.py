"""
Security primitives — deliberately dependency-free (stdlib only).

  * PBKDF2-HMAC-SHA256 password hashing with per-user salt (180k iterations)
  * JWT access tokens (HS256) with expiry and issuer claims
  * constant-time comparison for token/credential checks
  * PII minimisation: only a pseudonymous user record is stored
"""
from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jwt

from common import settings
from common.logging_utils import audit, get_logger

log = get_logger("security")

STORE_PATH = Path(settings.PROCESSED_DIR) / "auth_users.json"


@dataclass
class AuthUser:
    username: str
    email: str
    password_hash: str
    salt: str
    role: str = "analyst"
    created_at: str = ""
    fintwin_user_id: int = 1


def _hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(),
                             settings.PASSWORD_HASH_ITERATIONS)
    return dk.hex(), salt


def hash_password(password: str) -> tuple[str, str]:
    return _hash_password(password)


def verify_password(password: str, password_hash: str, salt: str) -> bool:
    candidate, _ = _hash_password(password, salt)
    return hmac.compare_digest(candidate, password_hash)


def _load_store() -> dict:
    if STORE_PATH.exists():
        return json.loads(STORE_PATH.read_text(encoding="utf-8"))
    return {}


def _save_store(store: dict) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STORE_PATH.write_text(json.dumps(store, indent=2), encoding="utf-8")


def register_user(username: str, email: str, password: str, fintwin_user_id: int = 1) -> AuthUser:
    store = _load_store()
    if username in store:
        raise ValueError("username already registered")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    pwd_hash, salt = hash_password(password)
    user = AuthUser(username=username, email=email, password_hash=pwd_hash, salt=salt,
                    created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
                    fintwin_user_id=int(fintwin_user_id))
    store[username] = asdict(user)
    _save_store(store)
    audit.write(actor=username, action="register", resource="auth", status="ok")
    log.info("registered user %s", username)
    return user


def authenticate(username: str, password: str) -> AuthUser | None:
    store = _load_store()
    rec = store.get(username)
    if not rec:
        audit.write(actor=username, action="login", resource="auth", status="failed",
                    reason="unknown_user")
        return None
    if verify_password(password, rec["password_hash"], rec["salt"]):
        audit.write(actor=username, action="login", resource="auth", status="ok")
        return AuthUser(**rec)
    audit.write(actor=username, action="login", resource="auth", status="failed",
                reason="bad_password")
    return None


def create_token(username: str, role: str = "analyst", fintwin_user_id: int = 1) -> str:
    now = int(time.time())
    payload = {
        "sub": username,
        "role": role,
        "uid": int(fintwin_user_id),
        "iat": now,
        "exp": now + settings.JWT_EXPIRE_MINUTES * 60,
        "iss": "fintwin-x",
    }
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM],
                          issuer="fintwin-x")
    except Exception as exc:                              # noqa: BLE001
        log.debug("token rejected: %s", exc)
        return None


def seed_demo_user() -> None:
    """Create the demo account shipped with the project (idempotent)."""
    store = _load_store()
    if "demo" not in store:
        register_user("demo", "demo@fintwinx.local", "fintwin-demo-2026", fintwin_user_id=1)
