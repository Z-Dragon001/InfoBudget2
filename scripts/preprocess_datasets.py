"""功能：批量预处理 LOCOMO 和 LongMemEval。
输入：configs 中的数据集配置与 raw 文件。
输出：datasets/processed 下的统一 JSONL。
依赖：项目配置与数据集预处理模块。
作者：OpenAI Codex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from infobudget.config import load_project_bundle
from infobudget.datasets.preprocess import DatasetPreprocessManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess LOCOMO and LongMemEval datasets.")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to preprocess.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_project_bundle("configs")
    manager = DatasetPreprocessManager(bundle.config.dataset, bundle.root_dir)
    summary = manager.preprocess_all(datasets=args.datasets or bundle.config.dataset.supported_datasets)
    for dataset_name, splits in summary.items():
        print(f"{dataset_name}: {splits}")


if __name__ == "__main__":
    main()
