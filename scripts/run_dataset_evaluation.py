"""功能：执行预处理后数据集的 mock 评估。
输入：processed 数据集文件与项目配置。
输出：按数据集与 split 写出的 metrics/predictions。
依赖：项目配置与 dataset runner。
作者：OpenAI Codex
"""

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
    parser = argparse.ArgumentParser(description="Run dataset evaluation for InfoBudget.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to evaluate.")
    parser.add_argument("--splits", nargs="*", default=None, help="Optional splits to evaluate.")
    parser.add_argument("--sample-ids", nargs="*", default=None, help="Optional sample ids to evaluate.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of samples per split.")
    parser.add_argument(
        "--scoring-mode",
        choices=SCORING_MODES,
        default="full",
        help="Score used for threshold routing. Defaults to full intrinsic/utility fusion.",
    )
    parser.add_argument(
        "--extraction-mode",
        choices=["flat", "event"],
        default=None,
        help="Memory extraction mode. Defaults to configs/config.yaml.",
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
    runner = DatasetEvaluationRunner(bundle, args.scoring_mode)
    loader = DatasetLoader(bundle.config.dataset, bundle.root_dir)
    dataset_names = args.datasets or bundle.config.evaluation.datasets
    requested_splits = set(args.splits or [])
    for dataset_name in dataset_names:
        available_splits = loader.list_available_splits(dataset_name)
        splits = [split for split in available_splits if not requested_splits or split in requested_splits]
        for split in splits:
            try:
                result = runner.evaluate(
                    dataset_name,
                    split,
                    sample_ids=set(args.sample_ids) if args.sample_ids else None,
                    limit=args.limit,
                )
            except FileNotFoundError:
                continue
            print(
                f"{result.dataset_name}/{result.split}: "
                f"accuracy={result.metrics.accuracy:.4f} "
                f"cost={result.metrics.total_cost_usd:.6f}"
            )


if __name__ == "__main__":
    main()
