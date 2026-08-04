"""Run one configured BERT/TextTiling method over frozen processed data."""

from __future__ import annotations

import argparse
from dataclasses import replace

from infobudget.config import load_project_bundle
from infobudget.segmentation.pipeline import SegmentationRun


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("split")
    parser.add_argument("--method", choices=["nsp_text_tiling", "bert_mlp_text_tiling"])
    args = parser.parse_args()
    bundle = load_project_bundle("configs")
    if args.method:
        bundle.config.segmentation = replace(bundle.config.segmentation, method=args.method)
    result = SegmentationRun(bundle).run(args.dataset, args.split)
    print(result)


if __name__ == "__main__":
    main()
