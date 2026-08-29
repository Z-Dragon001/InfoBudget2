"""CLI for building LoCoMo/LongMemEval frozen reference Fact sets."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infobudget.config import load_project_bundle
from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.ledger import SqliteLedger, atomic_write_json
from reference_fact_pipeline.config import load_reference_config
from reference_fact_pipeline.io import export_reference_jsonl, load_topic_segments
from reference_fact_pipeline.pipeline import ReferenceFactPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build strong candidate-independent frozen reference Facts."
    )
    parser.add_argument("--segments", type=Path, required=True)
    parser.add_argument("--dataset", choices=("locomo", "longmemeval"), required=True)
    parser.add_argument("--project-config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--pipeline-config",
        type=Path,
        default=Path(__file__).with_name("config.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")

    config = load_reference_config(args.pipeline_config)
    bundle = load_project_bundle(args.project_config_dir)
    roles = {
        config.reference_extractor_role,
        config.coverage_extractor_role,
        config.grounding_judge_role,
    }
    require_api_keys(bundle.models, roles, operation="frozen reference Fact construction")
    run_id = args.run_id or datetime.now(timezone.utc).strftime("reference_%Y%m%dT%H%M%SZ")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        parser.error("--run-id may contain only letters, digits, dot, underscore and hyphen")

    segments = [
        item
        for item in load_topic_segments(args.segments)
        if item.dataset_name.lower() == args.dataset
        and (not args.sample_id or item.sample_id in set(args.sample_id))
    ]
    if args.limit is not None:
        segments = segments[: args.limit]
    if not segments:
        raise ValueError("no topic segments remain after dataset/sample filters")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = SqliteLedger(
        args.output_dir / "reference_facts.sqlite3",
        "reference_fact_sets",
        key_fields=("run_id", "segment_id", "source_content_hash", "config_hash"),
    )
    existing_rows = ledger.read_all()
    completed = {
        (
            str(row.get("run_id")),
            str(row.get("segment_id")),
            str(row.get("source_content_hash")),
            str(row.get("config_hash")),
        )
        for row in existing_rows
    }
    client = OpenAICompatibleClient(
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
    )
    pipeline = ReferenceFactPipeline(
        config=config,
        models=bundle.models,
        prices=bundle.prices,
        client=client,
        prompt_dir=Path(__file__).with_name("prompts"),
        raw_archive_dir=args.output_dir / "raw_responses" / run_id,
    )
    skipped = 0
    built = 0
    for segment in segments:
        key = (run_id, segment.segment_id, segment.source_content_hash, config.canonical_hash())
        if key in completed:
            if not args.resume:
                raise RuntimeError(
                    f"segment already exists for run {run_id}; pass --resume or choose a new run-id: "
                    f"{segment.segment_id}"
                )
            skipped += 1
            continue
        result = pipeline.process_segment(segment, run_id=run_id)
        if not ledger.append(result.to_dict()):
            raise RuntimeError(f"concurrent duplicate reference result: {segment.segment_id}")
        built += 1

    requested_segment_ids = {item.segment_id for item in segments}
    selected_rows = [
        row
        for row in ledger.read_all()
        if row.get("run_id") == run_id and row.get("config_hash") == config.canonical_hash()
        and row.get("segment_id") in requested_segment_ids
    ]
    output = export_reference_jsonl(args.output_dir / "reference_facts.jsonl", selected_rows)
    manifest = _manifest(
        run_id=run_id,
        config_hash=config.canonical_hash(),
        dataset=args.dataset,
        source=args.segments,
        output=output,
        rows=selected_rows,
        built=built,
        skipped=skipped,
        model_roles={role: bundle.models[role].effective_model_name for role in sorted(roles)},
    )
    atomic_write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))


def _manifest(
    *,
    run_id: str,
    config_hash: str,
    dataset: str,
    source: Path,
    output: Path,
    rows: list[dict[str, Any]],
    built: int,
    skipped: int,
    model_roles: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": "frozen_reference_manifest_v1",
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset": dataset,
        "segments_source": str(source.resolve()),
        "reference_facts_output": str(output.resolve()),
        "segment_count": len(rows),
        "fact_count": sum(int(row.get("frozen_fact_count", 0)) for row in rows),
        "total_cost": sum(float(row.get("total_cost", 0.0)) for row in rows),
        "built_this_invocation": built,
        "skipped_as_completed": skipped,
        "model_roles": model_roles,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    main()
