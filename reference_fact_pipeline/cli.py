"""CLI for building LoCoMo/LongMemEval frozen reference Fact sets."""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from infobudget.config import load_project_bundle
from infobudget.rl_router.api import ModelAPIError
from reference_fact_pipeline.cloudflare_api import (
    CloudflareWholesaleRateLimitError,
    build_reference_api_client,
    require_reference_api_credentials,
)
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
    require_reference_api_credentials(
        bundle.models, roles, operation="frozen reference Fact construction"
    )
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

    client = build_reference_api_client(
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        retry_backoff_seconds=config.retry_backoff_seconds,
        cloudflare_request_interval_seconds=config.cloudflare_request_interval_seconds,
        cloudflare_402_backoff_seconds=config.cloudflare_402_backoff_seconds,
    )
    pipeline = ReferenceFactPipeline(
        config=config,
        models=bundle.models,
        prices=bundle.prices,
        client=client,
        prompt_dir=Path(__file__).with_name("prompts"),
        candidate_prompt_dir=bundle.prompt_dir,
        raw_archive_dir=args.output_dir / "raw_responses" / run_id,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ledger = SqliteLedger(
        args.output_dir / "reference_facts.sqlite3",
        "reference_fact_sets",
        key_fields=("run_id", "segment_id", "source_content_hash", "config_hash"),
    )
    failure_ledger = SqliteLedger(
        args.output_dir / "reference_facts.sqlite3",
        "reference_fact_failures",
        key_fields=("run_id", "segment_id", "source_content_hash", "config_hash"),
    )
    existing_rows = ledger.read_all()
    existing_failures = {
        (
            str(row.get("run_id")),
            str(row.get("segment_id")),
            str(row.get("source_content_hash")),
            str(row.get("config_hash")),
        ): row
        for row in failure_ledger.read_all()
    }
    completed = {
        (
            str(row.get("run_id")),
            str(row.get("segment_id")),
            str(row.get("source_content_hash")),
            str(row.get("config_hash")),
        )
        for row in existing_rows
    }
    skipped = 0
    built = 0
    failed = 0
    run_paused = False
    pause_reason = ""
    remaining_segment_count = 0
    circuit_pause_count = 0
    consecutive_auto_resumes = 0
    for segment_index, segment in enumerate(segments):
        key = (run_id, segment.segment_id, segment.source_content_hash, pipeline.effective_config_hash)
        if key in completed:
            if not args.resume:
                raise RuntimeError(
                    f"segment already exists for run {run_id}; pass --resume or choose a new run-id: "
                    f"{segment.segment_id}"
                )
            previous_failure = existing_failures.get(key)
            if previous_failure is not None and previous_failure.get("status") == "failed":
                failure_ledger.upsert(
                    {
                        **previous_failure,
                        "status": "resolved",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            skipped += 1
            continue
        while True:
            try:
                result = pipeline.process_segment(segment, run_id=run_id)
                if not ledger.append(result.to_dict()):
                    raise RuntimeError(
                        f"concurrent duplicate reference result: {segment.segment_id}"
                    )
                built += 1
                previous_failure = existing_failures.get(key)
                if previous_failure is not None:
                    resolved = {
                        **previous_failure,
                        "status": "resolved",
                        "resolved_at": datetime.now(timezone.utc).isoformat(),
                    }
                    failure_ledger.upsert(resolved)
                    existing_failures[key] = resolved
                consecutive_auto_resumes = 0
                break
            except (ModelAPIError, ValueError) as error:
                previous_failure = existing_failures.get(key, {})
                failure_row = {
                    "schema_version": "reference_fact_failure_v1",
                    "run_id": run_id,
                    "dataset": segment.dataset_name,
                    "split": segment.split,
                    "sample_id": segment.sample_id,
                    "segment_id": segment.segment_id,
                    "source_content_hash": segment.source_content_hash,
                    "config_hash": pipeline.effective_config_hash,
                    "status": "failed",
                    "attempt_count": int(previous_failure.get("attempt_count", 0)) + 1,
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                    "error_type": type(error).__name__,
                    "error": str(error)[:4000],
                    "transport_attempts": getattr(error, "attempts", None) or [],
                }
                failure_ledger.upsert(failure_row)
                existing_failures[key] = failure_row
                if isinstance(error, CloudflareWholesaleRateLimitError):
                    if consecutive_auto_resumes < config.cloudflare_max_auto_resumes:
                        consecutive_auto_resumes += 1
                        circuit_pause_count += 1
                        time.sleep(config.cloudflare_circuit_pause_seconds)
                        continue
                    failed += 1
                    run_paused = True
                    pause_reason = "persistent_cloudflare_wholesale_rate_limit"
                    remaining_segment_count = len(segments) - segment_index - 1
                    break
                failed += 1
                consecutive_auto_resumes = 0
                break
        if run_paused:
            break

    requested_segment_ids = {item.segment_id for item in segments}
    selected_rows = [
        row
        for row in ledger.read_all()
        if row.get("run_id") == run_id and row.get("config_hash") == pipeline.effective_config_hash
        and row.get("segment_id") in requested_segment_ids
    ]
    selected_failure_rows = [
        row
        for row in failure_ledger.read_all()
        if row.get("run_id") == run_id
        and row.get("config_hash") == pipeline.effective_config_hash
        and row.get("segment_id") in requested_segment_ids
        and row.get("status") == "failed"
    ]
    output = export_reference_jsonl(args.output_dir / "reference_facts.jsonl", selected_rows)
    manifest = _manifest(
        run_id=run_id,
        config_hash=pipeline.effective_config_hash,
        dataset=args.dataset,
        source=args.segments,
        output=output,
        rows=selected_rows,
        built=built,
        skipped=skipped,
        failed=failed,
        failure_rows=selected_failure_rows,
        run_paused=run_paused,
        pause_reason=pause_reason,
        remaining_segment_count=remaining_segment_count,
        circuit_pause_count=circuit_pause_count,
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
    failed: int = 0,
    failure_rows: list[dict[str, Any]] | None = None,
    run_paused: bool = False,
    pause_reason: str = "",
    remaining_segment_count: int = 0,
    circuit_pause_count: int = 0,
) -> dict[str, Any]:
    failures = failure_rows or []
    total_input_tokens = sum(int(row.get("total_input_tokens", 0)) for row in rows)
    total_output_tokens = sum(int(row.get("total_output_tokens", 0)) for row in rows)
    usage_by_role: dict[str, dict[str, int]] = {}
    for row in rows:
        for usage in row.get("stage_usage", ()):
            if not isinstance(usage, dict):
                continue
            role = str(usage.get("role") or "unknown")
            aggregate = usage_by_role.setdefault(
                role,
                {
                    "api_calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
            )
            input_tokens = int(usage.get("input_tokens", 0))
            output_tokens = int(usage.get("output_tokens", 0))
            aggregate["api_calls"] += 1
            aggregate["input_tokens"] += input_tokens
            aggregate["output_tokens"] += output_tokens
            aggregate["total_tokens"] += input_tokens + output_tokens
    return {
        "schema_version": "frozen_reference_manifest_v1",
        "run_id": run_id,
        "config_hash": config_hash,
        "dataset": dataset,
        "segments_source": str(source.resolve()),
        "reference_facts_output": str(output.resolve()),
        "segment_count": len(rows),
        "fact_count": sum(int(row.get("frozen_fact_count", 0)) for row in rows),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens": total_input_tokens + total_output_tokens,
        "provider_usage_stage_count": sum(
            int(row.get("provider_usage_stage_count", 0)) for row in rows
        ),
        "estimated_usage_stage_count": sum(
            int(row.get("estimated_usage_stage_count", 0)) for row in rows
        ),
        "token_usage_by_role": usage_by_role,
        "total_cost": sum(float(row.get("total_cost", 0.0)) for row in rows),
        "cost_complete": all(bool(row.get("cost_complete", True)) for row in rows),
        "unknown_cost_stage_count": sum(
            int(row.get("unknown_cost_stage_count", 0)) for row in rows
        ),
        "built_this_invocation": built,
        "skipped_as_completed": skipped,
        "failed_this_invocation": failed,
        "unresolved_failure_count": len(failures),
        "failed_segment_ids": sorted(str(row.get("segment_id")) for row in failures),
        "run_paused": run_paused,
        "pause_reason": pause_reason,
        "remaining_segment_count": remaining_segment_count,
        "circuit_pause_count": circuit_pause_count,
        "run_complete": len(failures) == 0 and not run_paused,
        "model_roles": model_roles,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


if __name__ == "__main__":
    main()
