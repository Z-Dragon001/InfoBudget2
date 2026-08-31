"""Deterministic local epsilon-noninferiority routing."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Iterable

from infobudget.quality_router.schemas import FactSetKey, QualityPrediction


_FALLBACK_ACTIONS = {"predicted_best"}


@dataclass(frozen=True, slots=True)
class QualityGapPolicy:
    epsilon: float
    quality_floor: float = 0.0
    uncertainty_enabled: bool = False
    gap_residual_bound: float = 0.0
    confidence: float = 0.95
    low_quality_action: str = "predicted_best"
    ood_action: str = "predicted_best"

    def __post_init__(self) -> None:
        for name, value in (
            ("epsilon", self.epsilon),
            ("quality_floor", self.quality_floor),
            ("gap_residual_bound", self.gap_residual_bound),
        ):
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.epsilon > 1.0 or self.quality_floor > 1.0:
            raise ValueError("epsilon and quality_floor must be in [0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must be in (0, 1)")
        if self.low_quality_action not in _FALLBACK_ACTIONS:
            raise ValueError(f"unsupported low_quality_action: {self.low_quality_action}")
        if self.ood_action not in _FALLBACK_ACTIONS:
            raise ValueError(f"unsupported ood_action: {self.ood_action}")

    def to_dict(self) -> dict:
        return {
            "epsilon": self.epsilon,
            "quality_floor": self.quality_floor,
            "uncertainty": {
                "enabled": self.uncertainty_enabled,
                "method": "validation_gap_residual_quantile",
                "confidence": self.confidence,
                "gap_residual_bound": self.gap_residual_bound,
            },
            "low_quality_action": self.low_quality_action,
            "ood_action": self.ood_action,
        }

    @classmethod
    def from_dict(cls, value: dict) -> "QualityGapPolicy":
        uncertainty = value.get("uncertainty") or {}
        if not isinstance(uncertainty, dict):
            raise ValueError("uncertainty must be an object")
        return cls(
            epsilon=float(value["epsilon"]),
            quality_floor=float(value.get("quality_floor", 0.0)),
            uncertainty_enabled=bool(uncertainty.get("enabled", False)),
            gap_residual_bound=float(uncertainty.get("gap_residual_bound", 0.0)),
            confidence=float(uncertainty.get("confidence", 0.95)),
            low_quality_action=str(value.get("low_quality_action") or "predicted_best"),
            ood_action=str(value.get("ood_action") or "predicted_best"),
        )


@dataclass(frozen=True, slots=True)
class QualityGapDecision:
    key: FactSetKey
    selected: QualityPrediction
    best: QualityPrediction
    best_predicted_quality: float
    eligible_model_ids: tuple[str, ...]
    predicted_gaps: dict[str, float]
    gap_upper_bounds: dict[str, float]
    decision_reason: str
    segment_ood: bool

    def to_dict(self) -> dict:
        return {
            "dataset": self.key.dataset,
            "split": self.key.split,
            "sample_id": self.key.sample_id,
            "segment_id": self.key.segment_id,
            "selected_model_id": self.selected.model_id,
            "selected_profile_id": self.selected.profile_id,
            "predicted_quality": self.selected.predicted_quality,
            "selected_cost": self.selected.cost,
            "best_model_id": self.best.model_id,
            "best_predicted_quality": self.best_predicted_quality,
            "eligible_model_ids": list(self.eligible_model_ids),
            "predicted_gap": dict(sorted(self.predicted_gaps.items())),
            "gap_upper_bound": dict(sorted(self.gap_upper_bounds.items())),
            "decision_reason": self.decision_reason,
            "segment_ood": self.segment_ood,
        }


def select_quality_gap_model(
    predictions: Iterable[QualityPrediction],
    *,
    policy: QualityGapPolicy,
    segment_ood: bool = False,
) -> QualityGapDecision:
    candidates = list(predictions)
    if not candidates:
        raise ValueError("quality-gap routing requires at least one candidate")
    key = candidates[0].key
    model_ids: set[str] = set()
    for candidate in candidates:
        if candidate.key != key:
            raise ValueError("quality-gap candidates must belong to one segment")
        if candidate.model_id in model_ids:
            raise ValueError(f"duplicate model candidate: {candidate.model_id}")
        model_ids.add(candidate.model_id)
        if not isfinite(candidate.predicted_quality) or not 0.0 <= candidate.predicted_quality <= 1.0:
            raise ValueError("predicted quality must be finite and in [0, 1]")
        if not isfinite(candidate.cost) or candidate.cost < 0.0:
            raise ValueError("candidate cost must be finite and non-negative")

    best_quality = max(candidate.predicted_quality for candidate in candidates)
    best = min(
        (candidate for candidate in candidates if abs(candidate.predicted_quality - best_quality) <= 1e-12),
        key=lambda candidate: (candidate.cost, candidate.model_id),
    )
    predicted_gaps = {
        candidate.model_id: max(0.0, best_quality - candidate.predicted_quality)
        for candidate in candidates
    }
    residual = policy.gap_residual_bound if policy.uncertainty_enabled else 0.0
    upper_bounds = {
        model_id: gap + residual for model_id, gap in predicted_gaps.items()
    }

    if segment_ood:
        selected = _fallback(best, policy.ood_action)
        eligible = (selected.model_id,)
        reason = "ood_predicted_best"
    elif best_quality < policy.quality_floor:
        selected = _fallback(best, policy.low_quality_action)
        eligible = (selected.model_id,)
        reason = "below_quality_floor_predicted_best"
    else:
        eligible_candidates = [
            candidate
            for candidate in candidates
            if upper_bounds[candidate.model_id] <= policy.epsilon + 1e-12
        ]
        if eligible_candidates:
            selected = min(
                eligible_candidates,
                key=lambda candidate: (
                    candidate.cost,
                    -candidate.predicted_quality,
                    candidate.model_id,
                ),
            )
            eligible = tuple(sorted(candidate.model_id for candidate in eligible_candidates))
            reason = "cheapest_model_within_quality_tolerance"
        else:
            selected = best
            eligible = (best.model_id,)
            reason = "no_robust_candidate_predicted_best"

    return QualityGapDecision(
        key=key,
        selected=selected,
        best=best,
        best_predicted_quality=best_quality,
        eligible_model_ids=eligible,
        predicted_gaps=predicted_gaps,
        gap_upper_bounds=upper_bounds,
        decision_reason=reason,
        segment_ood=segment_ood,
    )


def _fallback(best: QualityPrediction, action: str) -> QualityPrediction:
    if action == "predicted_best":
        return best
    raise AssertionError(f"validated fallback action is unsupported: {action}")
