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
    args = parser.parse_args()
    manifest = build_fact_equivalence_pairs(
        segments_path=args.segments,
        references_path=args.references,
        candidates_path=args.candidates,
        output_path=args.output,
        manifest_path=args.manifest_output,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
