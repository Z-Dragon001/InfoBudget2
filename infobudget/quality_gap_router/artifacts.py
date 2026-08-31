"""Strict artifact joins for quality-gap calibration and evaluation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from infobudget.quality_gap_router.schemas import QualityObservation, group_observations
from infobudget.quality_router.io import iter_jsonl
from infobudget.quality_router.schemas import (
    FactQualityLabel,
    FactSetKey,
    QualityPrediction,
)


def load_costs(path: str | Path) -> tuple[dict[tuple, float], dict[tuple, str]]:
    costs: dict[tuple, float] = {}
    tiers: dict[tuple, str] = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row)
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        if not model_id:
            raise ValueError("cost row is missing model_id/extractor_model")
        cost = float(row.get("cost", row.get("allocated_total_cost", -1)))
        if cost < 0.0:
            raise ValueError("cost row is missing a non-negative cost")
        identity = (key.tuple(), model_id)
        if identity in costs:
            raise ValueError(f"duplicate segment/model cost: {identity}")
        costs[identity] = cost
        tier = str(row.get("tier") or row.get("memory_tier") or "").strip()
        if tier:
            if tier not in {"small", "medium", "large"}:
                raise ValueError("cost row tier must be small, medium, or large")
            tiers[identity] = tier
    if not costs:
        raise ValueError("no cost rows found")
    return costs, tiers


def load_predictions(path: str | Path) -> dict[tuple, dict]:
    predictions: dict[tuple, dict] = {}
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row)
        model_id = str(row.get("model_id") or row.get("extractor_model") or "").strip()
        profile_id = str(row.get("profile_id") or "").strip()
        if not model_id or not profile_id:
            raise ValueError("prediction row is missing model_id/profile_id")
        identity = (key.tuple(), model_id)
        if identity in predictions:
            raise ValueError(f"duplicate quality prediction: {identity}")
        predictions[identity] = {
            "key": key,
            "model_id": model_id,
            "profile_id": profile_id,
            "predicted_quality": float(row["predicted_quality"]),
        }
    if not predictions:
        raise ValueError("no prediction rows found")
    return predictions


def load_labels(path: str | Path) -> dict[tuple, FactQualityLabel]:
    labels: dict[tuple, FactQualityLabel] = {}
    for row in iter_jsonl(path):
        label = FactQualityLabel.from_dict(row)
        identity = (label.key.tuple(), label.model_id)
        if identity in labels:
            raise ValueError(f"duplicate quality label: {identity}")
        labels[identity] = label
    if not labels:
        raise ValueError("no quality labels found")
    return labels


def join_observations(
    *,
    predictions: dict[tuple, dict],
    labels: dict[tuple, FactQualityLabel],
    costs: dict[tuple, float],
) -> dict[FactSetKey, tuple[QualityObservation, ...]]:
    prediction_keys = set(predictions)
    label_keys = set(labels)
    if prediction_keys != label_keys:
        raise ValueError(
            "prediction/label identities differ; "
            f"missing_predictions={sorted(label_keys - prediction_keys)[:10]}, "
            f"missing_labels={sorted(prediction_keys - label_keys)[:10]}"
        )
    missing_costs = sorted(prediction_keys - set(costs))
    if missing_costs:
        raise ValueError(f"costs are missing prediction identities: {missing_costs[:10]}")
    observations: list[QualityObservation] = []
    for identity in sorted(prediction_keys):
        value = predictions[identity]
        label = labels[identity]
        if value["profile_id"] != label.profile_id:
            raise ValueError(f"prediction/label profile mismatch: {identity}")
        observations.append(
            QualityObservation(
                prediction=QualityPrediction(
                    key=value["key"],
                    model_id=value["model_id"],
                    profile_id=value["profile_id"],
                    predicted_quality=value["predicted_quality"],
                    cost=costs[identity],
                ),
                actual_quality=label.silver_strict_fact_f1,
            )
        )
    return group_observations(observations)


def group_predictions(
    predictions: list[QualityPrediction],
) -> dict[FactSetKey, tuple[QualityPrediction, ...]]:
    grouped: dict[FactSetKey, list[QualityPrediction]] = defaultdict(list)
    seen: set[tuple] = set()
    for prediction in predictions:
        identity = (prediction.key.tuple(), prediction.model_id)
        if identity in seen:
            raise ValueError(f"duplicate quality prediction: {identity}")
        seen.add(identity)
        grouped[prediction.key].append(prediction)
    return {
        key: tuple(sorted(rows, key=lambda row: row.model_id))
        for key, rows in grouped.items()
    }
