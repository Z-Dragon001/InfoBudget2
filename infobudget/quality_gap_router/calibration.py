"""Validation-only residual calibration and epsilon selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from infobudget.quality_gap_router.decision import QualityGapPolicy
from infobudget.quality_gap_router.evaluation import (
    QualityGapEvaluation,
    evaluate_quality_gap,
)
from infobudget.quality_gap_router.schemas import QualityObservation
from infobudget.quality_router.schemas import FactSetKey


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    policy: QualityGapPolicy
    selected_evaluation: QualityGapEvaluation
    sweep: tuple[dict, ...]
    validation_segment_count: int
    constraints_satisfied: bool

    def to_dict(self) -> dict:
        return {
            "schema_version": "quality_gap_calibration_v1",
            "decision_mode": "quality_gap",
            **self.policy.to_dict(),
            "validation_segment_count": self.validation_segment_count,
            "constraints_satisfied": self.constraints_satisfied,
            "selected_metrics": self.selected_evaluation.metrics_dict(),
        }


def estimate_gap_residual_bound(
    groups: dict[FactSetKey, tuple[QualityObservation, ...]],
    *,
    confidence: float,
) -> float:
    if not groups:
        raise ValueError("cannot calibrate an empty validation set")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    residuals: list[float] = []
    for observations in groups.values():
        predicted_best = max(row.prediction.predicted_quality for row in observations)
        actual_best = max(row.actual_quality for row in observations)
        for row in observations:
            predicted_gap = predicted_best - row.prediction.predicted_quality
            actual_gap = actual_best - row.actual_quality
            residuals.append(actual_gap - predicted_gap)
    return max(
        0.0,
        float(np.quantile(np.asarray(residuals, dtype=np.float64), confidence, method="higher")),
    )


def calibrate_quality_gap(
    groups: dict[FactSetKey, tuple[QualityObservation, ...]],
    *,
    epsilon_values: list[float],
    quality_floor: float,
    uncertainty_enabled: bool,
    confidence: float,
    violation_threshold: float,
    max_mean_regret: float,
    max_violation_rate: float,
) -> CalibrationResult:
    if not epsilon_values:
        raise ValueError("epsilon_values cannot be empty")
    epsilon_values = sorted(set(float(value) for value in epsilon_values))
    if any(value < 0.0 or value > 1.0 for value in epsilon_values):
        raise ValueError("epsilon values must be in [0, 1]")
    if max_mean_regret < 0.0 or not 0.0 <= max_violation_rate <= 1.0:
        raise ValueError("invalid calibration constraints")
    residual_bound = (
        estimate_gap_residual_bound(groups, confidence=confidence)
        if uncertainty_enabled
        else 0.0
    )
    evaluations: list[tuple[QualityGapPolicy, QualityGapEvaluation]] = []
    sweep: list[dict] = []
    for epsilon in epsilon_values:
        policy = QualityGapPolicy(
            epsilon=epsilon,
            quality_floor=quality_floor,
            uncertainty_enabled=uncertainty_enabled,
            gap_residual_bound=residual_bound,
            confidence=confidence,
        )
        evaluation = evaluate_quality_gap(
            groups,
            policy=policy,
            violation_threshold=violation_threshold,
        )
        feasible = (
            evaluation.mean_regret <= max_mean_regret + 1e-12
            and evaluation.violation_rate <= max_violation_rate + 1e-12
        )
        evaluations.append((policy, evaluation))
        sweep.append(
            {
                "epsilon": epsilon,
                "constraints_satisfied": feasible,
                **evaluation.metrics_dict(),
            }
        )
    feasible_rows = [
        (policy, evaluation)
        for policy, evaluation in evaluations
        if evaluation.mean_regret <= max_mean_regret + 1e-12
        and evaluation.violation_rate <= max_violation_rate + 1e-12
    ]
    if not feasible_rows:
        raise ValueError(
            "no epsilon satisfies the declared validation constraints; "
            "relax the constraints or improve/calibrate the quality scorer"
        )
    selected_policy, selected_evaluation = max(
        feasible_rows,
        key=lambda item: (
            item[0].epsilon,
            item[1].cost_saving_vs_highest_candidate,
            -item[1].mean_regret,
        ),
    )
    return CalibrationResult(
        policy=selected_policy,
        selected_evaluation=selected_evaluation,
        sweep=tuple(sweep),
        validation_segment_count=len(groups),
        constraints_satisfied=True,
    )
