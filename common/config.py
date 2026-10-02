"""
FinTwin-X — central configuration.

Every module imports `settings` from here so paths, seeds and business
constants are defined exactly once (single source of truth).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

# --------------------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------------------
ROOT: Path = Path(__file__).resolve().parents[1]


def _load_local_env(path: Path) -> None:
    """Load simple KEY=VALUE pairs for local runs without overriding process secrets."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.replace("_", "").isalnum():
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))


_load_local_env(ROOT / ".env")


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw) if raw else default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    return default if raw is None else int(raw)


@dataclass(frozen=True)
class Settings:
    # ---------------------------------------------------------------- filesystem
    ROOT: Path = ROOT
    DATA_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_DATA_DIR", ROOT / "data"))
    RAW_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_RAW_DIR", ROOT / "data" / "raw"))
    PROCESSED_DIR: Path = field(
        default_factory=lambda: _env_path("FINTWIN_PROCESSED_DIR", ROOT / "data" / "processed")
    )
    SYNTHETIC_DIR: Path = field(
        default_factory=lambda: _env_path("FINTWIN_SYNTHETIC_DIR", ROOT / "data" / "synthetic")
    )
    WAREHOUSE_DIR: Path = field(
        default_factory=lambda: _env_path("FINTWIN_WAREHOUSE_DIR", ROOT / "data" / "warehouse")
    )
    EXPORT_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_EXPORT_DIR", ROOT / "data" / "exports"))
    MODEL_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_MODEL_DIR", ROOT / "ml" / "models" / "artifacts"))
    REGISTRY_DIR: Path = field(
        default_factory=lambda: _env_path("FINTWIN_REGISTRY_DIR", ROOT / "ml" / "models" / "registry")
    )
    VECTOR_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_VECTOR_DIR", ROOT / "rag" / "vector_store"))
    LOG_DIR: Path = field(default_factory=lambda: _env_path("FINTWIN_LOG_DIR", ROOT / "logs"))

    # ---------------------------------------------------------------- simulation clock
    # The synthetic "today". Everything is generated up to (and including) this month.
    AS_OF: date = field(default_factory=lambda: date.fromisoformat(os.getenv("AS_OF", "2026-08-31")))
    HISTORY_MONTHS: int = 24          # months of generated history
    FORECAST_HORIZONS: tuple = (1, 3, 6, 12)
    MAX_SIM_HORIZON: int = 120        # months

    # ---------------------------------------------------------------- domain constants
    CURRENCY: str = "PKR"
    CURRENCY_SYMBOL: str = "Rs"
    CITIES: tuple = (
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad",
        "Multan", "Peshawar", "Quetta", "Hyderabad", "Sialkot",
    )
    CITY_COST_INDEX: dict = field(
        default_factory=lambda: {
            "Karachi": 1.08, "Lahore": 1.02, "Islamabad": 1.12, "Rawalpindi": 0.98,
            "Faisalabad": 0.88, "Multan": 0.86, "Peshawar": 0.85, "Quetta": 0.92,
            "Hyderabad": 0.84, "Sialkot": 0.90,
        }
    )
    EXPENSE_CATEGORIES: tuple = (
        "Housing", "Food", "Transport", "Education", "Healthcare",
        "Entertainment", "Shopping", "Utilities", "Subscriptions", "Other",
    )
    ESSENTIAL_CATEGORIES: tuple = ("Housing", "Food", "Transport", "Education", "Healthcare", "Utilities")
    DISCRETIONARY_CATEGORIES: tuple = ("Entertainment", "Shopping", "Subscriptions", "Other")

    # ---------------------------------------------------------------- runtime
    ENVIRONMENT: str = os.getenv("FINTWIN_ENV", "dev").lower()
    RANDOM_SEED: int = field(default_factory=lambda: _env_int("RANDOM_SEED", 42))
    N_SIMULATIONS: int = field(default_factory=lambda: _env_int("N_SIMULATIONS", 10_000))
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = field(default_factory=lambda: _env_int("API_PORT", 8000))

    # ---------------------------------------------------------------- security
    JWT_SECRET: str = os.getenv("FINTWIN_JWT_SECRET", "dev-only-insecure-secret-change-me")
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = field(default_factory=lambda: _env_int("JWT_EXPIRE_MINUTES", 60 * 12))
    PASSWORD_HASH_ITERATIONS: int = field(default_factory=lambda: _env_int("FINTWIN_PASSWORD_HASH_ITERATIONS", 180_000))
    RATE_LIMIT_PER_MINUTE: int = field(default_factory=lambda: _env_int("RATE_LIMIT_PER_MINUTE", 240))
    ENABLE_DEMO_USER: bool = field(default_factory=lambda: _env_bool(
        "FINTWIN_ENABLE_DEMO_USER", os.getenv("FINTWIN_ENV", "dev").lower() not in {"prod", "production"}
    ))
    CORS_ORIGINS: tuple = field(default_factory=lambda: tuple(
        origin.strip() for origin in os.getenv(
            "FINTWIN_CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
        ).split(",") if origin.strip()
    ))

    # ---------------------------------------------------------------- external
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "")
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
    EMBEDDING_BACKEND: str = os.getenv("FINTWIN_EMBEDDING_BACKEND", "tfidf")  # tfidf | openai | sentence-transformers
    ENABLE_SHAP: bool = field(default_factory=lambda: _env_bool("FINTWIN_ENABLE_SHAP", True))

    @property
    def duckdb_path(self) -> Path:
        p = self.WAREHOUSE_DIR / "fintwin.duckdb"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()

if settings.ENVIRONMENT in {"prod", "production"} and settings.JWT_SECRET.startswith(("dev-only", "change-")):
    raise RuntimeError("FINTWIN_JWT_SECRET must be replaced before production startup")

for _d in (
    settings.DATA_DIR, settings.RAW_DIR, settings.PROCESSED_DIR, settings.SYNTHETIC_DIR,
    settings.WAREHOUSE_DIR, settings.EXPORT_DIR, settings.MODEL_DIR, settings.REGISTRY_DIR,
    settings.VECTOR_DIR, settings.LOG_DIR,
):
    _d.mkdir(parents=True, exist_ok=True)
