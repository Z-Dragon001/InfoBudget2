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
    parser.add_argument(
        "--alpha",
        type=float,
        help="Override segmentation.adaptive_alpha; the value is embedded in the output directory name.",
    )
    args = parser.parse_args()
    bundle = load_project_bundle("configs")
    if args.method:
        bundle.config.segmentation = replace(bundle.config.segmentation, method=args.method)
    if args.alpha is not None:
        if args.alpha < 0:
            parser.error("--alpha must be non-negative")
        bundle.config.segmentation = replace(
            bundle.config.segmentation, adaptive_alpha=args.alpha
        )
    result = SegmentationRun(bundle).run(args.dataset, args.split)
    print(result)


if __name__ == "__main__":
    main()
