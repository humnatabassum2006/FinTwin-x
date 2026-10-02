"""Structured, dependency-free logging for FinTwin-X."""
from __future__ import annotations

import json
import logging
import sys
import threading
import time
from pathlib import Path

from common.config import settings

_LOCK = threading.Lock()
_CONFIGURED = False


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # noqa: D102
        payload = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def get_logger(name: str = "fintwin", level: int = logging.INFO) -> logging.Logger:
    """Return a logger with a human-readable console handler (and JSON file handler)."""
    global _CONFIGURED
    with _LOCK:
        if not _CONFIGURED:
            root = logging.getLogger("fintwin")
            root.setLevel(level)
            root.propagate = False
            console = logging.StreamHandler(sys.stdout)
            console.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s", "%H:%M:%S")
            )
            root.addHandler(console)
            try:
                fh = logging.FileHandler(settings.LOG_DIR / "fintwin.log", encoding="utf-8")
                fh.setFormatter(JsonFormatter())
                root.addHandler(fh)
            except Exception:  # pragma: no cover - read-only filesystem
                pass
            _CONFIGURED = True
    return logging.getLogger(f"fintwin.{name}") if name != "fintwin" else logging.getLogger("fintwin")


class AuditLog:
    """Append-only JSONL audit trail (who / what / when / outcome).

    A real deployment would ship this to an immutable store; here it is a
    newline-delimited JSON file that survives process restarts.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or settings.LOG_DIR / "audit.jsonl")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def write(self, *, actor: str, action: str, resource: str = "", status: str = "ok", **fields) -> None:
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "actor": actor,
            "action": action,
            "resource": resource,
            "status": status,
            **fields,
        }
        line = json.dumps(record, ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")

    def tail(self, n: int = 200) -> list[dict]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                continue
        return out


audit = AuditLog()
