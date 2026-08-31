"""Evaluate quality-gap decisions against held-out Strict Fact-F1 labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_gap_router.artifacts import (
    join_observations,
    load_costs,
    load_labels,
    load_predictions,
)
from infobudget.quality_gap_router.decision import QualityGapPolicy
from infobudget.quality_gap_router.evaluation import evaluate_quality_gap
from infobudget.quality_router.io import iter_jsonl, write_jsonl
from infobudget.quality_router.schemas import FactSetKey
from infobudget.rl_router.ledger import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--costs", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--ood-flags", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("schema_version") != "quality_gap_calibration_v1":
        raise ValueError("calibration schema_version must be quality_gap_calibration_v1")
    policy = QualityGapPolicy.from_dict(calibration)
    constraints = calibration.get("selection_constraints") or {}
    violation_threshold = float(
        constraints.get("violation_threshold", policy.epsilon)
    )
    costs, _ = load_costs(args.costs)
    groups = join_observations(
        predictions=load_predictions(args.predictions),
        labels=load_labels(args.labels),
        costs=costs,
    )
    evaluation = evaluate_quality_gap(
        groups,
        policy=policy,
        violation_threshold=violation_threshold,
        ood_keys=_load_ood_flags(args.ood_flags) if args.ood_flags else None,
    )
    atomic_write_json(args.output, evaluation.metrics_dict())
    write_jsonl(args.rows_output, evaluation.rows)
    print(
        json.dumps(
            {
                **evaluation.metrics_dict(),
                "output": str(args.output.resolve()),
                "rows_output": str(args.rows_output.resolve()),
            },
            ensure_ascii=False,
        )
    )


def _load_ood_flags(path: Path) -> set[FactSetKey]:
    result: set[FactSetKey] = set()
    for row in iter_jsonl(path):
        key = FactSetKey.from_dict(row)
        if bool(row.get("segment_ood", row.get("is_ood", row.get("ood", False)))):
            if key in result:
                raise ValueError(f"duplicate OOD flag: {key.tuple()}")
            result.add(key)
    return result


if __name__ == "__main__":
    main()
