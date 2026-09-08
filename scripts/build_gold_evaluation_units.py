"""Normalize reviewed Gold Facts into frozen evaluation claim units."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from infobudget.quality_router.segment_set_evaluation import (
    plan_gold_evaluation_units,
    run_gold_evaluation_units,
)
from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.config import load_rl_bundle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path("configs/prompts/gold_evaluation_unit_builder_v1.txt"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.max_segments is not None and args.max_segments <= 0:
        parser.error("--max-segments must be positive")
    bundle = load_rl_bundle(args.config_dir)
    model = bundle.project.models["judge_llm"]
    price = bundle.project.prices[model.model_name]
    if args.plan_only:
        result = plan_gold_evaluation_units(
            references_path=args.references, prompt_path=args.prompt,
            model_spec=model, price=price,
        )
    else:
        require_api_keys(bundle.project.models, ("judge_llm",), operation="Gold evaluation-unit normalization")
        reliability = bundle.rl.get("api_reliability", {})
        client = OpenAICompatibleClient(
            timeout_seconds=int(reliability.get("timeout_seconds", 300)),
            max_retries=int(reliability.get("max_retries", 4)),
            retry_backoff_seconds=float(reliability.get("retry_backoff_seconds", 1.0)),
        )
        result = run_gold_evaluation_units(
            references_path=args.references, prompt_path=args.prompt,
            output_dir=args.output_dir, output_path=args.output,
            model_spec=model, price=price, client=client,
            max_segments=args.max_segments,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
