"""Judge anonymous Candidate Fact sets against frozen Gold claim units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.segment_set_evaluation import (
    plan_segment_set_judging,
    run_segment_set_judging,
)
from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.config import load_rl_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--candidate-inventory", type=Path, required=True)
    parser.add_argument(
        "--prompt", type=Path,
        default=Path("configs/prompts/segment_fact_set_judge_v3.txt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--anonymization-seed", type=int, default=42)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--semantic-retries", type=int, default=2)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_segments is not None and args.max_segments <= 0:
        parser.error("--max-segments must be positive")
    if args.semantic_retries < 0 or args.semantic_retries > 5:
        parser.error("--semantic-retries must be between 0 and 5")
    bundle = load_rl_bundle(args.config_dir)
    model = bundle.project.models["judge_llm"]
    price = bundle.project.prices[model.model_name]
    common = dict(
        segments_path=args.segments, references_path=args.references,
        reference_manifest_path=args.reference_manifest,
        candidates_path=args.candidates,
        candidate_inventory_path=args.candidate_inventory,
        prompt_path=args.prompt,
        model_spec=model, price=price,
        anonymization_seed=args.anonymization_seed,
    )
    if args.plan_only:
        result = plan_segment_set_judging(**common)
    else:
        require_api_keys(bundle.project.models, ("judge_llm",), operation="segment-level Candidate-set judging")
        reliability = bundle.rl.get("api_reliability", {})
        client = OpenAICompatibleClient(
            timeout_seconds=int(reliability.get("timeout_seconds", 300)),
            max_retries=int(reliability.get("max_retries", 4)),
            retry_backoff_seconds=float(reliability.get("retry_backoff_seconds", 1.0)),
        )
        result = run_segment_set_judging(
            **common, output_dir=args.output_dir, output_path=args.output,
            client=client, max_segments=args.max_segments,
            semantic_retries=args.semantic_retries,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
