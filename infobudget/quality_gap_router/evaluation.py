"""Held-out Fact-quality evaluation for local quality-gap routing."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from infobudget.quality_gap_router.decision import (
    QualityGapPolicy,
    select_quality_gap_model,
)
from infobudget.quality_gap_router.schemas import QualityObservation
from infobudget.quality_router.schemas import FactSetKey


@dataclass(frozen=True, slots=True)
class QualityGapEvaluation:
    segment_count: int
    mean_selected_quality: float
    mean_oracle_quality: float
    mean_regret: float
    median_regret: float
    p90_regret: float
    p95_regret: float
    violation_threshold: float
    violation_rate: float
    total_selected_cost: float
    total_highest_candidate_cost: float
    cost_saving_vs_highest_candidate: float
    true_epsilon_oracle_agreement: float
    selection_counts: dict[str, int]
    decision_reason_counts: dict[str, int]
    rows: tuple[dict, ...]

    def metrics_dict(self) -> dict:
        return {
            "schema_version": "quality_gap_evaluation_v1",
            "segment_count": self.segment_count,
            "mean_selected_quality": self.mean_selected_quality,
            "mean_oracle_quality": self.mean_oracle_quality,
            "mean_regret": self.mean_regret,
            "median_regret": self.median_regret,
            "p90_regret": self.p90_regret,
            "p95_regret": self.p95_regret,
            "violation_threshold": self.violation_threshold,
            "violation_rate": self.violation_rate,
            "total_selected_cost": self.total_selected_cost,
            "total_highest_candidate_cost": self.total_highest_candidate_cost,
            "cost_saving_vs_highest_candidate": self.cost_saving_vs_highest_candidate,
            "true_epsilon_oracle_agreement": self.true_epsilon_oracle_agreement,
            "selection_counts": dict(sorted(self.selection_counts.items())),
            "decision_reason_counts": dict(sorted(self.decision_reason_counts.items())),
        }


def evaluate_quality_gap(
    groups: dict[FactSetKey, tuple[QualityObservation, ...]],
    *,
    policy: QualityGapPolicy,
    violation_threshold: float,
    ood_keys: set[FactSetKey] | None = None,
) -> QualityGapEvaluation:
    if not groups:
        raise ValueError("quality-gap evaluation groups are empty")
    if not 0.0 <= violation_threshold <= 1.0:
        raise ValueError("violation_threshold must be in [0, 1]")
    ood_keys = ood_keys or set()
    rows: list[dict] = []
    regrets: list[float] = []
    selected_qualities: list[float] = []
    oracle_qualities: list[float] = []
    selected_costs: list[float] = []
    highest_costs: list[float] = []
    oracle_agreements: list[bool] = []
    selection_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()

    for key in sorted(groups, key=lambda item: item.tuple()):
        observations = groups[key]
        by_model = {row.model_id: row for row in observations}
        decision = select_quality_gap_model(
            [row.prediction for row in observations],
            policy=policy,
            segment_ood=key in ood_keys,
        )
        selected = by_model[decision.selected.model_id]
        oracle_quality = max(row.actual_quality for row in observations)
        regret = max(0.0, oracle_quality - selected.actual_quality)
        true_eligible = [
            row
            for row in observations
            if oracle_quality - row.actual_quality <= policy.epsilon + 1e-12
        ]
        true_epsilon_oracle = min(
            true_eligible,
            key=lambda row: (
                row.prediction.cost,
                -row.actual_quality,
                row.model_id,
            ),
        )
        highest_cost = max(row.prediction.cost for row in observations)

        regrets.append(regret)
        selected_qualities.append(selected.actual_quality)
        oracle_qualities.append(oracle_quality)
        selected_costs.append(selected.prediction.cost)
        highest_costs.append(highest_cost)
        oracle_agreements.append(selected.model_id == true_epsilon_oracle.model_id)
        selection_counts[selected.model_id] += 1
        reason_counts[decision.decision_reason] += 1
        rows.append(
            {
                **decision.to_dict(),
                "actual_selected_quality": selected.actual_quality,
                "actual_oracle_quality": oracle_quality,
                "actual_regret": regret,
                "true_epsilon_oracle_model_id": true_epsilon_oracle.model_id,
                "true_epsilon_oracle_agreement": selected.model_id
                == true_epsilon_oracle.model_id,
                "quality_violation": regret > violation_threshold + 1e-12,
            }
        )

    regret_array = np.asarray(regrets, dtype=np.float64)
    total_selected_cost = float(sum(selected_costs))
    total_highest_cost = float(sum(highest_costs))
    saving = (
        1.0 - total_selected_cost / total_highest_cost
        if total_highest_cost > 0.0
        else 0.0
    )
    return QualityGapEvaluation(
        segment_count=len(rows),
        mean_selected_quality=float(np.mean(selected_qualities)),
        mean_oracle_quality=float(np.mean(oracle_qualities)),
        mean_regret=float(np.mean(regret_array)),
        median_regret=float(np.quantile(regret_array, 0.50)),
        p90_regret=float(np.quantile(regret_array, 0.90)),
        p95_regret=float(np.quantile(regret_array, 0.95)),
        violation_threshold=violation_threshold,
        violation_rate=float(np.mean(regret_array > violation_threshold + 1e-12)),
        total_selected_cost=total_selected_cost,
        total_highest_candidate_cost=total_highest_cost,
        cost_saving_vs_highest_candidate=saving,
        true_epsilon_oracle_agreement=float(np.mean(oracle_agreements)),
        selection_counts=dict(selection_counts),
        decision_reason_counts=dict(reason_counts),
        rows=tuple(rows),
    )
