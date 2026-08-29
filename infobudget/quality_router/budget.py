"""Deterministic multiple-choice budget allocation over segment/model scores."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

from infobudget.quality_router.schemas import FactSetKey, QualityPrediction


@dataclass(frozen=True, slots=True)
class BudgetSolution:
    selections: tuple[QualityPrediction, ...]
    total_quality: float
    total_cost: float
    budget: float


def optimize_budget(
    predictions: Iterable[QualityPrediction],
    *,
    budget: float,
    cost_quantum: float = 0.000001,
) -> BudgetSolution:
    """Maximize summed predicted quality with one model selected per segment.

    Costs are rounded upward to integer units, so a returned solution never exceeds
    the real-valued budget. Dominated dynamic-programming states are pruned after
    every segment.
    """
    if budget < 0:
        raise ValueError("budget cannot be negative")
    if cost_quantum <= 0:
        raise ValueError("cost_quantum must be positive")
    grouped: dict[FactSetKey, list[QualityPrediction]] = {}
    for prediction in predictions:
        if prediction.cost < 0:
            raise ValueError("candidate cost cannot be negative")
        if not 0.0 <= prediction.predicted_quality <= 1.0:
            raise ValueError("predicted quality must be in [0, 1]")
        grouped.setdefault(prediction.key, []).append(prediction)
    if not grouped:
        return BudgetSolution((), 0.0, 0.0, budget)
    for key, candidates in grouped.items():
        model_ids = [item.model_id for item in candidates]
        if len(model_ids) != len(set(model_ids)):
            raise ValueError(f"duplicate model candidate for segment: {key.tuple()}")

    budget_units = math.floor((budget + 1e-12) / cost_quantum)
    # cost units -> (quality, selections). One state before the first segment.
    states: dict[int, tuple[float, tuple[QualityPrediction, ...]]] = {0: (0.0, ())}
    for key in sorted(grouped, key=lambda value: value.tuple()):
        next_states: dict[int, tuple[float, tuple[QualityPrediction, ...]]] = {}
        candidates = sorted(grouped[key], key=lambda item: item.model_id)
        for used_units, (quality, chosen) in states.items():
            for candidate in candidates:
                candidate_units = math.ceil((candidate.cost - 1e-15) / cost_quantum)
                total_units = used_units + max(0, candidate_units)
                if total_units > budget_units:
                    continue
                candidate_quality = quality + candidate.predicted_quality
                previous = next_states.get(total_units)
                if previous is None or candidate_quality > previous[0] + 1e-12:
                    next_states[total_units] = (candidate_quality, chosen + (candidate,))
        if not next_states:
            raise ValueError(
                "budget is below the minimum feasible one-model-per-segment cost"
            )
        states = _prune_dominated(next_states)

    used_units, (total_quality, selections) = max(
        states.items(), key=lambda item: (item[1][0], -item[0])
    )
    del used_units
    total_cost = sum(item.cost for item in selections)
    if total_cost > budget + 1e-9:
        raise AssertionError("internal budget rounding allowed an infeasible solution")
    return BudgetSolution(selections, total_quality, total_cost, budget)


def _prune_dominated(
    states: dict[int, tuple[float, tuple[QualityPrediction, ...]]]
) -> dict[int, tuple[float, tuple[QualityPrediction, ...]]]:
    result: dict[int, tuple[float, tuple[QualityPrediction, ...]]] = {}
    best_quality = float("-inf")
    for cost in sorted(states):
        value = states[cost]
        if value[0] > best_quality + 1e-12:
            result[cost] = value
            best_quality = value[0]
    return result
