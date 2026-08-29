"""Build artifact B from A, or summarize prejoined counterfactual artifact C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.counterfactual import (
    aggregate_segment_usage,
    counterfactual_consistency,
)
from infobudget.quality_router.io import iter_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    usage = subparsers.add_parser("usage")
    usage.add_argument("--trace", type=Path, required=True)
    usage.add_argument("--output", type=Path, required=True)
    consistency = subparsers.add_parser("consistency")
    consistency.add_argument("--counterfactuals", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "usage":
        rows = aggregate_segment_usage(iter_jsonl(args.trace))
        write_jsonl(args.output, rows)
        print(json.dumps({"segments": len(rows), "output": str(args.output.resolve())}, ensure_ascii=False))
    else:
        print(json.dumps(counterfactual_consistency(iter_jsonl(args.counterfactuals)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
