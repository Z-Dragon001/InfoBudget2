"""Merge audited candidate Fact exports into one model-keyed JSONL corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.candidate_corpus import build_candidate_fact_corpus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-root", type=Path, default=Path("outputs/rl_router/runs"))
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--campaign-id", action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inventory-output", type=Path, required=True)
    args = parser.parse_args()

    inventory = build_candidate_fact_corpus(
        runs_root=args.runs_root,
        segments_path=args.segments,
        campaign_ids=args.campaign_id,
        output_path=args.output,
        inventory_path=args.inventory_output,
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "candidate_fact_count": inventory["candidate_fact_count"],
                "models": inventory["models"],
                "output": str(args.output.resolve()),
                "inventory_output": str(args.inventory_output.resolve()),
                "sha256": inventory["candidate_facts_sha256"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
