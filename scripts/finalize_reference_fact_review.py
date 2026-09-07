"""Finalize a post-review Reference Fact audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from reference_fact_pipeline.manual_review import finalize_completed_review_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconcile post-review audit triggers with completed decisions."
    )
    parser.add_argument("--automated-audit", type=Path, required=True)
    parser.add_argument("--post-review-queue", type=Path, required=True)
    parser.add_argument("--original-review-queue", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--application-summary", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_completed_review_audit(
        automated_audit_path=args.automated_audit,
        post_review_queue_path=args.post_review_queue,
        original_review_queue_path=args.original_review_queue,
        decisions_path=args.decisions,
        application_summary_path=args.application_summary,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
