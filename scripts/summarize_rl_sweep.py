"""Summarize completed alpha/epoch routed experiments for paper tables."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from infobudget.rl_router.ledger import atomic_write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=("locomo", "longmemeval"))
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--method", help="Optional exact parameterized segmentation method.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    experiments_root = (
        root
        / "outputs"
        / "rl_router"
        / "full_experiments"
        / args.dataset
        / args.protocol
    )
    pattern = (
        f"{args.method}/epochs_*/*/manifest.json"
        if args.method
        else "*/epochs_*/*/manifest.json"
    )
    rows = []
    for path in sorted(experiments_root.glob(pattern)):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        aggregate = manifest.get("aggregate") or {}
        if manifest.get("status") != "complete" or not aggregate:
            continue
        rows.append(
            {
                "experiment_id": manifest["experiment_id"],
                "dataset_name": manifest["dataset_name"],
                "protocol": manifest.get("protocol"),
                "segmentation_method": manifest["segmentation_method"],
                "adaptive_alpha": manifest.get("adaptive_alpha"),
                "epochs": int(manifest["epochs"]),
                "fold_count": int(aggregate.get("fold_count", 0)),
                "question_count": int(aggregate.get("question_count", 0)),
                "qa_accuracy_micro": float(aggregate.get("qa_accuracy_micro", 0.0)),
                "mean_fold_accuracy_micro": float(
                    aggregate.get("mean_fold_accuracy_micro", 0.0)
                ),
                "std_fold_accuracy_micro": float(
                    aggregate.get("std_fold_accuracy_micro", 0.0)
                ),
                "sem_fold_accuracy_micro": float(
                    aggregate.get("sem_fold_accuracy_micro", 0.0)
                ),
                "end_to_end_known_cost": float(
                    aggregate.get("end_to_end_known_cost", 0.0)
                ),
                "full_experiment_known_cost": float(
                    aggregate.get("full_experiment_known_cost", 0.0)
                ),
                "manifest": str(path.resolve()),
            }
        )
    rows.sort(
        key=lambda row: (
            -row["mean_fold_accuracy_micro"],
            row["std_fold_accuracy_micro"],
            row["full_experiment_known_cost"],
        )
    )
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else experiments_root / "sweep_summary"
    )
    payload = {
        "schema_version": "rl_sweep_summary_v1",
        "dataset_name": args.dataset,
        "protocol": args.protocol,
        "selection_rule": (
            "descending mean fold QA accuracy, ascending fold accuracy standard deviation, "
            "ascending full known experiment cost"
        ),
        "experiment_count": len(rows),
        "experiments": rows,
    }
    atomic_write_json(output_dir / "summary.json", payload)
    _write_csv(output_dir / "summary.csv", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "rank",
        "experiment_id",
        "dataset_name",
        "protocol",
        "segmentation_method",
        "adaptive_alpha",
        "epochs",
        "fold_count",
        "question_count",
        "qa_accuracy_micro",
        "mean_fold_accuracy_micro",
        "std_fold_accuracy_micro",
        "sem_fold_accuracy_micro",
        "end_to_end_known_cost",
        "full_experiment_known_cost",
        "manifest",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
