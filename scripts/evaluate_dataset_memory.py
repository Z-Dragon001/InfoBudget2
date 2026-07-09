"""Evaluate QA against previously built dataset memory stores."""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infobudget.config import load_project_bundle
from infobudget.datasets.loader import DatasetLoader
from infobudget.evaluation.dataset_runner import DatasetEvaluationRunner
from infobudget.scoring.modes import SCORING_MODES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate InfoBudget using existing dataset memories only.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to evaluate.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional splits to evaluate.")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Optional sample ids to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples per split.")
    parser.add_argument(
        "--scoring-modes",
        nargs="*",
        choices=SCORING_MODES,
        default=["full"],
        help="Memory scoring-mode folders to evaluate. Defaults to full.",
    )
    parser.add_argument(
        "--all-scoring-modes",
        action="store_true",
        help="Evaluate memory stores for all 9 scoring modes.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["flat", "event"],
        default=None,
        help="Memory extraction mode folder to evaluate. Defaults to configs/config.yaml.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_project_bundle("configs")
    if args.extraction_mode:
        bundle = replace(
            bundle,
            config=replace(
                bundle.config,
                extractor=replace(bundle.config.extractor, extraction_mode=args.extraction_mode),
            ),
        )
    loader = DatasetLoader(bundle.config.dataset, bundle.root_dir)
    dataset_names = args.datasets or bundle.config.evaluation.datasets
    requested_splits = set(args.splits or [])
    scoring_modes = SCORING_MODES if args.all_scoring_modes else tuple(args.scoring_modes)
    for scoring_mode in scoring_modes:
        runner = DatasetEvaluationRunner(bundle, scoring_mode)
        for dataset_name in dataset_names:
            available_splits = loader.list_available_splits(dataset_name)
            splits = [split for split in available_splits if not requested_splits or split in requested_splits]
            for split in splits:
                try:
                    result = runner.evaluate_existing_memories(
                        dataset_name,
                        split,
                        sample_ids=set(args.sample_ids) if args.sample_ids else None,
                        limit=args.limit,
                    )
                except FileNotFoundError as exc:
                    print(f"skip {dataset_name}/{split}/{scoring_mode}: {exc}")
                    continue
                print(
                    f"evaluated {result.dataset_name}/{result.split}/{scoring_mode}: "
                    f"accuracy={result.metrics.accuracy:.4f} "
                    f"cost={result.metrics.total_cost_usd:.6f}"
                )


if __name__ == "__main__":
    main()
