"""Plan or run resumable grounding and directional-relation Fact judgments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.equivalence_judge import (
    plan_equivalence_judging,
    run_equivalence_judging,
)
from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.config import load_rl_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--pairs-manifest", type=Path)
    parser.add_argument(
        "--prompt",
        type=Path,
        default=Path("configs/prompts/fact_relation_judge_v2.txt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")

    bundle = load_rl_bundle(args.config_dir)
    model = bundle.project.models["judge_llm"]
    if model.backend != "openai_compatible":
        parser.error("judge_llm must use the openai_compatible backend")
    price = bundle.project.prices[model.model_name]
    if args.plan_only:
        result = plan_equivalence_judging(
            segments_path=args.segments,
            pairs_path=args.pairs,
            pairs_manifest_path=args.pairs_manifest,
            prompt_path=args.prompt,
            model_spec=model,
            price=price,
            batch_size=args.batch_size,
        )
    else:
        require_api_keys(
            bundle.project.models,
            ("judge_llm",),
            operation="Fact-equivalence judging",
        )
        reliability = bundle.rl.get("api_reliability", {})
        client = OpenAICompatibleClient(
            timeout_seconds=int(reliability.get("timeout_seconds", 300)),
            max_retries=int(reliability.get("max_retries", 4)),
            retry_backoff_seconds=float(
                reliability.get("retry_backoff_seconds", 1.0)
            ),
        )
        result = run_equivalence_judging(
            segments_path=args.segments,
            pairs_path=args.pairs,
            pairs_manifest_path=args.pairs_manifest,
            prompt_path=args.prompt,
            output_dir=args.output_dir,
            output_path=args.output,
            model_spec=model,
            price=price,
            client=client,
            batch_size=args.batch_size,
            max_batches=args.max_batches,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
