"""Build the frozen, source-overlapping pair universe for the Fact Judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.equivalence_pairs import build_fact_equivalence_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path, required=True)
    parser.add_argument(
        "--include-no-source-overlap",
        action="store_true",
        help=(
            "Include every candidate/Gold pair in the same segment and let the "
            "semantic Judge validate each side's cited evidence independently."
        ),
    )
    args = parser.parse_args()
    manifest = build_fact_equivalence_pairs(
        segments_path=args.segments,
        references_path=args.references,
        candidates_path=args.candidates,
        output_path=args.output,
        manifest_path=args.manifest_output,
        require_source_overlap=not args.include_no_source_overlap,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
