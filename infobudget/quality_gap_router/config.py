"""Configuration loader for quality-gap calibration defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class QualityGapRouterConfig:
    values: dict

    @classmethod
    def load(
        cls, path: str | Path = "configs/quality_gap_router.yaml"
    ) -> "QualityGapRouterConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if payload.get("schema_version") != "quality_gap_router_v1":
            raise ValueError("quality-gap schema_version must be quality_gap_router_v1")
        values = payload.get("quality_gap_router")
        if not isinstance(values, dict):
            raise ValueError("quality_gap_router configuration is missing")
        start = float(values.get("epsilon_grid_start", -1))
        end = float(values.get("epsilon_grid_end", -1))
        step = float(values.get("epsilon_grid_step", 0))
        if not 0.0 <= start <= end <= 1.0 or step <= 0.0:
            raise ValueError("invalid epsilon grid")
        for name in ("quality_floor", "violation_threshold", "max_mean_regret"):
            value = float(values.get(name, -1))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"quality_gap_router.{name} must be in [0, 1]")
        max_violation_rate = float(values.get("max_violation_rate", -1))
        confidence = float(values.get("confidence", -1))
        if not 0.0 <= max_violation_rate <= 1.0:
            raise ValueError("max_violation_rate must be in [0, 1]")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        return cls(dict(values))

    def epsilon_values(self) -> list[float]:
        start = float(self.values["epsilon_grid_start"])
        end = float(self.values["epsilon_grid_end"])
        step = float(self.values["epsilon_grid_step"])
        values: list[float] = []
        current = start
        while current <= end + step * 1e-6:
            values.append(round(current, 10))
            current += step
        return values
