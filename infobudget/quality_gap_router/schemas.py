"""Data contracts for local quality-gap routing and validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from infobudget.quality_router.schemas import FactSetKey, QualityPrediction


@dataclass(frozen=True, slots=True)
class QualityObservation:
    """A frozen prediction paired with its held-out realized Fact quality."""

    prediction: QualityPrediction
    actual_quality: float

    def __post_init__(self) -> None:
        if not isfinite(self.actual_quality) or not 0.0 <= self.actual_quality <= 1.0:
            raise ValueError("actual_quality must be finite and in [0, 1]")

    @property
    def key(self) -> FactSetKey:
        return self.prediction.key

    @property
    def model_id(self) -> str:
        return self.prediction.model_id


def group_observations(
    observations: list[QualityObservation],
) -> dict[FactSetKey, tuple[QualityObservation, ...]]:
    grouped: dict[FactSetKey, list[QualityObservation]] = {}
    seen: set[tuple[tuple[str, str, str, str], str]] = set()
    for observation in observations:
        identity = (observation.key.tuple(), observation.model_id)
        if identity in seen:
            raise ValueError(f"duplicate quality observation: {identity}")
        seen.add(identity)
        grouped.setdefault(observation.key, []).append(observation)
    if not grouped:
        raise ValueError("quality observations are empty")
    result: dict[FactSetKey, tuple[QualityObservation, ...]] = {}
    for key, rows in grouped.items():
        if len(rows) < 2:
            raise ValueError(
                f"quality-gap routing requires at least two candidates: {key.tuple()}"
            )
        result[key] = tuple(sorted(rows, key=lambda row: row.model_id))
    return result
