"""Audit unique-Fact coverage before paying for semantic pair judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.pair_coverage_audit import audit_fact_pair_coverage


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risky-pairs-output", type=Path, required=True)
    parser.add_argument("--lexical-threshold", type=float, default=0.35)
    parser.add_argument("--sequence-threshold", type=float, default=0.72)
    parser.add_argument("--adjacent-turn-distance", type=int, default=1)
    args = parser.parse_args()
    result = audit_fact_pair_coverage(
        candidates_path=args.candidates,
        references_path=args.references,
        pairs_path=args.pairs,
        output_path=args.output,
        risky_pairs_output_path=args.risky_pairs_output,
        lexical_threshold=args.lexical_threshold,
        sequence_threshold=args.sequence_threshold,
        adjacent_turn_distance=args.adjacent_turn_distance,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

