"""
Local model registry (MLflow-compatible interface, zero infrastructure).

Every training run writes:
    ml/models/registry/<name>/<version>/model.joblib
    ml/models/registry/<name>/<version>/manifest.json

and updates `registry/index.json`. When MLFLOW_TRACKING_URI is configured the
same run is *also* logged to MLflow, so the code is production-ready without
requiring a server for the demo.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import joblib

from common import settings
from common.logging_utils import get_logger

log = get_logger("registry")


@dataclass
class ModelManifest:
    name: str
    version: str
    algorithm: str
    task: str
    created_at: str
    params: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    features: list = field(default_factory=list)
    dataset: str = ""
    rows: int = 0
    artifact: str = ""
    stage: str = "staging"          # staging | production | archived
    notes: str = ""


class Registry:
    def __init__(self, root: Path | None = None):
        self.root = Path(root or settings.REGISTRY_DIR)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "index.json"
        self.index = self._load_index()

    def _load_index(self) -> dict:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {"models": {}}

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2), encoding="utf-8")

    def _next_version(self, name: str) -> str:
        versions = [m["version"] for m in self.index["models"].get(name, [])]
        n = len(versions)
        return f"v{n + 1}"

    def register(self, name: str, model: Any, *, algorithm: str, task: str, metrics: dict,
                 params: dict | None = None, features: list | None = None, dataset: str = "",
                 rows: int = 0, stage: str = "staging", notes: str = "") -> ModelManifest:
        version = self._next_version(name)
        dir_ = self.root / name / version
        dir_.mkdir(parents=True, exist_ok=True)
        artifact = dir_ / "model.joblib"
        joblib.dump(model, artifact)

        man = ModelManifest(
            name=name, version=version, algorithm=algorithm, task=task,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
            params=params or {}, metrics=metrics, features=features or [],
            dataset=dataset, rows=rows, artifact=str(artifact.relative_to(settings.ROOT)),
            stage=stage, notes=notes,
        )
        (dir_ / "manifest.json").write_text(json.dumps(asdict(man), indent=2, default=str), encoding="utf-8")
        self.index["models"].setdefault(name, []).append(asdict(man))
        self._save_index()

        if settings.MLFLOW_TRACKING_URI:
            self._log_to_mlflow(man)
        log.info("registered %s %s (%s) metrics=%s", name, version, algorithm,
                 {k: metrics[k] for k in list(metrics)[:3]})
        return man

    def _log_to_mlflow(self, man: ModelManifest) -> None:  # pragma: no cover
        try:
            import mlflow
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(man.name)
            with mlflow.start_run(run_name=f"{man.name}-{man.version}"):
                mlflow.log_params(man.params)
                mlflow.log_metrics({k: float(v) for k, v in man.metrics.items()
                                    if isinstance(v, (int, float))})
                mlflow.set_tag("algorithm", man.algorithm)
                mlflow.log_artifact(str(settings.ROOT / man.artifact))
        except Exception as exc:
            log.warning("mlflow logging failed: %s", exc)

    def promote(self, name: str, version: str) -> None:
        for m in self.index["models"].get(name, []):
            m["stage"] = "production" if m["version"] == version else "archived"
        self._save_index()
        log.info("promoted %s %s to production", name, version)

    def get(self, name: str, version: str | None = None) -> dict | None:
        models = self.index["models"].get(name, [])
        if not models:
            return None
        if version:
            return next((m for m in models if m["version"] == version), None)
        return next((m for m in models if m["stage"] == "production"), models[-1])

    def load(self, name: str, version: str | None = None):
        man = self.get(name, version)
        if man is None:
            return None
        return joblib.load(settings.ROOT / man["artifact"])

    def list_models(self) -> dict:
        return self.index["models"]


registry = Registry()
