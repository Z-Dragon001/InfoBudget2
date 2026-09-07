"""Apply completed Excel review decisions to frozen Reference Facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_fact_pipeline.manual_review import apply_completed_review


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a reviewed Reference Fact derivative from an Excel audit."
    )
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--review-queue", type=Path, required=True)
    parser.add_argument("--review-workbook", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--review-version", default="manual_review_v1")
    parser.add_argument(
        "--blank-means-pass",
        action="store_true",
        help="Map blank decisions to the category-specific pass decision.",
    )
    args = parser.parse_args()
    result = apply_completed_review(
        references_path=args.references,
        manifest_path=args.manifest,
        review_queue_path=args.review_queue,
        review_workbook_path=args.review_workbook,
        output_dir=args.output_dir,
        run_id=args.run_id,
        review_version=args.review_version,
        blank_means_pass=args.blank_means_pass,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
