"""Tests for epsilon-noninferiority quality-gap routing."""

from __future__ import annotations

import pytest

from infobudget.quality_gap_router.calibration import (
    calibrate_quality_gap,
    estimate_gap_residual_bound,
)
from infobudget.quality_gap_router.config import QualityGapRouterConfig
from infobudget.quality_gap_router.decision import (
    QualityGapPolicy,
    select_quality_gap_model,
)
from infobudget.quality_gap_router.evaluation import evaluate_quality_gap
from infobudget.quality_gap_router.schemas import QualityObservation, group_observations
from infobudget.quality_router.schemas import FactSetKey, QualityPrediction


def _prediction(
    key: FactSetKey,
    model_id: str,
    quality: float,
    cost: float,
) -> QualityPrediction:
    return QualityPrediction(
        key=key,
        model_id=model_id,
        profile_id=f"profile-{model_id}",
        predicted_quality=quality,
        cost=cost,
    )


def _candidates(qualities: tuple[float, float, float]) -> list[QualityPrediction]:
    key = FactSetKey("dataset", "test", "sample", "segment")
    return [
        _prediction(key, "small", qualities[0], 1.0),
        _prediction(key, "medium", qualities[1], 2.0),
        _prediction(key, "large", qualities[2], 4.0),
    ]


def test_close_quality_selects_cheapest_sufficient_model() -> None:
    decision = select_quality_gap_model(
        _candidates((0.70, 0.71, 0.75)),
        policy=QualityGapPolicy(epsilon=0.05),
    )
    assert decision.selected.model_id == "small"
    assert set(decision.eligible_model_ids) == {"small", "medium", "large"}
    assert decision.predicted_gaps["small"] == pytest.approx(0.05)


def test_intermediate_gap_selects_medium_instead_of_forcing_large() -> None:
    decision = select_quality_gap_model(
        _candidates((0.70, 0.71, 0.76)),
        policy=QualityGapPolicy(epsilon=0.05),
    )
    assert decision.selected.model_id == "medium"
    assert set(decision.eligible_model_ids) == {"medium", "large"}


def test_large_gap_selects_predicted_best_model() -> None:
    decision = select_quality_gap_model(
        _candidates((0.45, 0.62, 0.83)),
        policy=QualityGapPolicy(epsilon=0.05),
    )
    assert decision.selected.model_id == "large"


def test_quality_floor_and_ood_use_conservative_predicted_best_fallback() -> None:
    low_quality = select_quality_gap_model(
        _candidates((0.20, 0.21, 0.22)),
        policy=QualityGapPolicy(epsilon=0.05, quality_floor=0.30),
    )
    assert low_quality.selected.model_id == "large"
    assert low_quality.decision_reason == "below_quality_floor_predicted_best"

    ood = select_quality_gap_model(
        _candidates((0.70, 0.71, 0.75)),
        policy=QualityGapPolicy(epsilon=0.05),
        segment_ood=True,
    )
    assert ood.selected.model_id == "large"
    assert ood.decision_reason == "ood_predicted_best"


def test_uncertainty_bound_prevents_aggressive_downgrade() -> None:
    decision = select_quality_gap_model(
        _candidates((0.70, 0.71, 0.75)),
        policy=QualityGapPolicy(
            epsilon=0.05,
            uncertainty_enabled=True,
            gap_residual_bound=0.03,
        ),
    )
    assert decision.selected.model_id == "large"
    assert decision.eligible_model_ids == ("large",)
    assert decision.decision_reason == "cheapest_model_within_quality_tolerance"


def test_gap_residual_calibration_captures_underestimated_pairwise_gap() -> None:
    key = FactSetKey("dataset", "validation", "sample", "segment")
    groups = group_observations(
        [
            QualityObservation(_prediction(key, "small", 0.70, 1.0), 0.60),
            QualityObservation(_prediction(key, "large", 0.75, 4.0), 0.80),
        ]
    )
    assert estimate_gap_residual_bound(groups, confidence=0.95) == pytest.approx(0.15)


def test_calibration_selects_largest_epsilon_satisfying_regret_constraints() -> None:
    first = FactSetKey("dataset", "validation", "sample", "s1")
    second = FactSetKey("dataset", "validation", "sample", "s2")
    groups = group_observations(
        [
            QualityObservation(_prediction(first, "small", 0.70, 1.0), 0.70),
            QualityObservation(_prediction(first, "medium", 0.71, 2.0), 0.71),
            QualityObservation(_prediction(first, "large", 0.75, 4.0), 0.75),
            QualityObservation(_prediction(second, "small", 0.45, 1.0), 0.45),
            QualityObservation(_prediction(second, "medium", 0.62, 2.0), 0.62),
            QualityObservation(_prediction(second, "large", 0.83, 4.0), 0.83),
        ]
    )
    result = calibrate_quality_gap(
        groups,
        epsilon_values=[0.0, 0.05, 0.10],
        quality_floor=0.0,
        uncertainty_enabled=False,
        confidence=0.95,
        violation_threshold=0.05,
        max_mean_regret=0.03,
        max_violation_rate=0.0,
    )
    assert result.policy.epsilon == 0.10
    assert result.selected_evaluation.selection_counts == {"small": 1, "large": 1}
    assert result.selected_evaluation.mean_regret == pytest.approx(0.025)


def test_evaluation_reports_regret_cost_saving_and_true_oracle_agreement() -> None:
    key = FactSetKey("dataset", "test", "sample", "segment")
    groups = group_observations(
        [
            QualityObservation(_prediction(key, "small", 0.70, 1.0), 0.70),
            QualityObservation(_prediction(key, "medium", 0.71, 2.0), 0.71),
            QualityObservation(_prediction(key, "large", 0.75, 4.0), 0.75),
        ]
    )
    evaluation = evaluate_quality_gap(
        groups,
        policy=QualityGapPolicy(epsilon=0.05),
        violation_threshold=0.05,
    )
    assert evaluation.mean_regret == pytest.approx(0.05)
    assert evaluation.cost_saving_vs_highest_candidate == pytest.approx(0.75)
    assert evaluation.true_epsilon_oracle_agreement == 1.0
    assert evaluation.violation_rate == 0.0


def test_quality_gap_config_has_valid_epsilon_grid() -> None:
    config = QualityGapRouterConfig.load("configs/quality_gap_router.yaml")
    values = config.epsilon_values()
    assert values[0] == 0.0
    assert values[-1] == 0.15
    assert len(values) == 16
