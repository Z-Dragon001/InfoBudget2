"""Configuration loader for the supervised quality-router path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from infobudget.quality_router.schemas import CAPABILITY_DIMENSIONS


@dataclass(frozen=True, slots=True)
class QualityRouterConfig:
    values: dict[str, Any]
    artifacts: dict[str, str]

    @classmethod
    def load(cls, path: str | Path = "configs/quality_router.yaml") -> "QualityRouterConfig":
        source = Path(path)
        payload = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if payload.get("schema_version") != "quality_router_v1":
            raise ValueError("quality router schema_version must be quality_router_v1")
        values = payload.get("quality_router")
        artifacts = payload.get("artifacts")
        if not isinstance(values, dict) or not isinstance(artifacts, dict):
            raise ValueError("quality router config is missing quality_router/artifacts")
        if tuple(values.get("capability_dimensions", ())) != CAPABILITY_DIMENSIONS:
            raise ValueError("configured capability dimensions do not match MemoryPrint v1")
        if values.get("label_name") != "silver_strict_fact_f1":
            raise ValueError("quality router label_name must be silver_strict_fact_f1")
        if any(int(item) <= 0 for item in values.get("hidden_dimensions", ())):
            raise ValueError("hidden_dimensions must be positive")
        for name in ("learning_rate", "batch_size", "epochs", "early_stopping_patience", "huber_delta", "budget_cost_quantum"):
            if float(values.get(name, 0)) <= 0:
                raise ValueError(f"quality_router.{name} must be positive")
        if not 0.0 <= float(values.get("dropout", -1)) < 1.0:
            raise ValueError("quality_router.dropout must be in [0, 1)")
        if float(values.get("weight_decay", -1)) < 0:
            raise ValueError("quality_router.weight_decay cannot be negative")
        return cls(dict(values), {str(key): str(value) for key, value in artifacts.items()})
