"""Calibrate a validation-only epsilon-noninferiority routing policy."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from infobudget.quality_gap_router.artifacts import (
    join_observations,
    load_costs,
    load_labels,
    load_predictions,
)
from infobudget.quality_gap_router.calibration import calibrate_quality_gap
from infobudget.quality_gap_router.config import QualityGapRouterConfig
from infobudget.quality_router.io import file_sha256, write_jsonl
from infobudget.rl_router.ledger import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/quality_gap_router.yaml")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep-output", type=Path, required=True)
    parser.add_argument("--epsilon-values", help="Comma-separated override, e.g. 0,0.01,0.05")
    parser.add_argument("--quality-floor", type=float)
    parser.add_argument("--confidence", type=float)
    parser.add_argument("--violation-threshold", type=float)
    parser.add_argument("--max-mean-regret", type=float)
    parser.add_argument("--max-violation-rate", type=float)
    parser.add_argument(
        "--disable-uncertainty", action="store_true", help="Use point-estimate gaps only."
    )
    args = parser.parse_args()

    config = QualityGapRouterConfig.load(args.config)
    values = config.values
    epsilon_values = (
        _parse_epsilon_values(args.epsilon_values)
        if args.epsilon_values
        else config.epsilon_values()
    )
    costs, _ = load_costs(args.costs)
    groups = join_observations(
        predictions=load_predictions(args.predictions),
        labels=load_labels(args.labels),
        costs=costs,
    )
    quality_floor = _value(args.quality_floor, values, "quality_floor")
    confidence = _value(args.confidence, values, "confidence")
    violation_threshold = _value(
        args.violation_threshold, values, "violation_threshold"
    )
    max_mean_regret = _value(args.max_mean_regret, values, "max_mean_regret")
    max_violation_rate = _value(
        args.max_violation_rate, values, "max_violation_rate"
    )
    uncertainty_enabled = bool(values.get("uncertainty_enabled", True)) and not args.disable_uncertainty
    result = calibrate_quality_gap(
        groups,
        epsilon_values=epsilon_values,
        quality_floor=quality_floor,
        uncertainty_enabled=uncertainty_enabled,
        confidence=confidence,
        violation_threshold=violation_threshold,
        max_mean_regret=max_mean_regret,
        max_violation_rate=max_violation_rate,
    )
    payload = {
        **result.to_dict(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_constraints": {
            "max_mean_regret": max_mean_regret,
            "violation_threshold": violation_threshold,
            "max_violation_rate": max_violation_rate,
        },
        "validation_artifacts": {
            "predictions_sha256": file_sha256(args.predictions),
            "labels_sha256": file_sha256(args.labels),
            "costs_sha256": file_sha256(args.costs),
            "config_sha256": file_sha256(args.config),
        },
    }
    atomic_write_json(args.output, payload)
    write_jsonl(args.sweep_output, result.sweep)
    print(
        json.dumps(
            {
                "epsilon": result.policy.epsilon,
                "gap_residual_bound": result.policy.gap_residual_bound,
                "segments": result.validation_segment_count,
                "output": str(args.output.resolve()),
                "sweep_output": str(args.sweep_output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _parse_epsilon_values(value: str) -> list[float]:
    values = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError("--epsilon-values contains no values")
    return values


def _value(override: float | None, values: dict, name: str) -> float:
    return float(values[name] if override is None else override)


if __name__ == "__main__":
    main()
