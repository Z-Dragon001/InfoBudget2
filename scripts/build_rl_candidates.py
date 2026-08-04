"""Build or resume one immutable, fully audited L/M/H candidate extraction run."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.campaign import (
    campaign_manifest_path,
    refresh_campaign,
    validate_campaign_environment,
)
from infobudget.rl_router.candidates import (
    CandidateGenerator,
    ProviderCircuitOpenError,
    estimate_candidate_plan,
    prepare_extraction_segments,
)
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.embedding import LocalSentenceEncoder, LocalTokenizer
from infobudget.rl_router.export import export_memories
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.manifest import (
    create_experiment_manifest,
    file_sha256,
    memory_embedding_hash,
    resolve_collection_namespace,
    safe_qdrant_storage_config,
    update_experiment_manifest,
)
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.schemas import BatchCompletion, ProviderUsage, TIERS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("segments_jsonl", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--campaign-id")
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--extraction-run-id", help="Create a new run with this ID.")
    run_group.add_argument("--resume", metavar="RUN_ID", help="Resume an existing run.")
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        choices=TIERS,
        help="Extract only this tier. Repeat to select multiple tiers; default is all tiers.",
    )
    parser.add_argument(
        "--retry-terminal",
        action="store_true",
        help="With --resume, explicitly retry terminal schema failures.",
    )
    args = parser.parse_args()
    if args.retry_terminal and not args.resume:
        parser.error("--retry-terminal requires --resume")

    bundle = load_rl_bundle(args.config_dir)
    selected_tiers = tuple(dict.fromkeys(args.tiers or TIERS))
    models = {tier: bundle.project.models[tier] for tier in TIERS}
    require_api_keys(
        models,
        selected_tiers,
        operation="candidate extraction",
    )
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    segment_path = args.segments_jsonl.resolve()
    segments = load_segments(segment_path)
    first = segments[0]
    embedding_cfg = bundle.embeddings["memory"]
    root = bundle.project.root_dir
    storage_cfg = bundle.rl["storage"]
    campaign = None
    if args.campaign_id:
        campaign_path = campaign_manifest_path(root, args.campaign_id)
        if not campaign_path.is_file():
            raise FileNotFoundError(f"campaign manifest is missing: {campaign_path}")
        campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    encoder = LocalSentenceEncoder(
        model_name=embedding_cfg["model_name"],
        local_path=root / embedding_cfg["local_path"],
        dimension=embedding_cfg["dimension"],
        normalize=embedding_cfg["normalize"],
    )
    smoke_vectors = encoder.encode(["InfoBudget embedding preflight."])
    if smoke_vectors.shape != (1, encoder.dimension):
        raise ValueError(f"embedding smoke test returned {smoke_vectors.shape}")
    tokenizers = {
        tier: LocalTokenizer(root / models[tier].tokenizer_local_path)
        for tier in selected_tiers
    }
    counters = {tier: tokenizer.count for tier, tokenizer in tokenizers.items()}
    if any(counter("InfoBudget tokenizer preflight.") <= 0 for counter in counters.values()):
        raise ValueError("tokenizer smoke test returned no tokens")
    embedding_hash = (
        str(campaign["embedding_model_hash"])
        if campaign is not None
        else memory_embedding_hash(bundle)
    )
    if campaign is not None:
        validate_campaign_environment(
            bundle, campaign, precomputed_embedding_hash=embedding_hash
        )
    namespace = resolve_collection_namespace(
        storage_cfg,
        dataset=first.dataset_name,
        split=first.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=embedding_hash,
    )
    schema_probe = FactQdrantStore.from_storage_config(
        storage_cfg,
        project_root=root,
        namespace=namespace,
    )
    schema_probe.close()
    shared_prompt = bundle.prompt_path("fact_extraction").read_text(encoding="utf-8")
    prompts = {tier: shared_prompt for tier in models}
    prompt_versions = {tier: "joint_memory_extraction_batch_json_v6" for tier in models}
    prices = {tier: bundle.project.prices[models[tier].model_name] for tier in models}
    output_root = root / "outputs" / "rl_router"
    run_id = args.resume or args.extraction_run_id or str(uuid.uuid4())
    run_dir = output_root / "runs" / run_id
    manifest_path = run_dir / "manifest.json"
    resume = bool(args.resume)
    if args.campaign_id:
        assert campaign is not None
        expected_run_id = (campaign.get("expected_runs") or {}).get(first.sample_id)
        if expected_run_id != run_id:
            raise ValueError(
                f"campaign expects run {expected_run_id!r} for sample {first.sample_id}, got {run_id!r}"
            )
        expected_sha = (campaign.get("sample_segment_sha256") or {}).get(first.sample_id)
        if expected_sha != file_sha256(segment_path):
            raise ValueError("segments file differs from the immutable campaign scope")

    if resume:
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"cannot resume run without manifest: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("extraction_run_id") != run_id:
            raise ValueError("resume manifest extraction_run_id mismatch")
        if manifest.get("segments_jsonl_sha256") != file_sha256(segment_path):
            raise ValueError("resume segments_jsonl differs from the original run")
        if manifest.get("qdrant_storage") != safe_qdrant_storage_config(storage_cfg):
            raise ValueError("resume Qdrant storage differs from the original run")
        if manifest.get("qdrant_collection_namespace") != namespace:
            raise ValueError("resume Qdrant collection namespace differs from the original run")
        if manifest.get("embedding_model_hash") != embedding_hash:
            raise ValueError("resume embedding model hash differs from the original run")
        if campaign and (
            manifest.get("campaign_id") != args.campaign_id
            or manifest.get("campaign_scope_hash") != campaign.get("campaign_scope_hash")
        ):
            raise ValueError("resume run does not belong to the selected campaign")
    else:
        if manifest_path.exists():
            raise ValueError(f"run already exists; use --resume {run_id}")
        manifest = create_experiment_manifest(
            bundle,
            manifest_path,
            experiment_id=run_id,
            run_type="candidate_extraction",
            extraction_run_id=run_id,
            status="planned",
            dataset_name=first.dataset_name,
            split=first.split,
            sample_id=first.sample_id,
            segmentation_method=first.segmentation_method,
            segmentation_version=first.segmentation_version,
            qdrant_collection_namespace=namespace,
            segments_jsonl=str(segment_path),
            segments_jsonl_sha256=file_sha256(segment_path),
            segment_count=len(segments),
            segment_ids=[segment.segment_id for segment in segments],
            required_tiers=list(TIERS),
            completed_tiers=[],
            campaign_id=args.campaign_id or "",
            campaign_scope_hash=(campaign or {}).get("campaign_scope_hash", ""),
            segment_content_hashes={
                segment.segment_id: segment.source_content_hash for segment in segments
            },
            precomputed_embedding_hash=embedding_hash,
        )
    extraction_segments, selected_truncation_plans = prepare_extraction_segments(
        segments=segments,
        prompts=prompts,
        extraction_config=bundle.rl["extraction"],
        token_counters=counters,
        tiers=selected_tiers,
    )
    selected_plan = estimate_candidate_plan(
        segments=extraction_segments,
        prompts={tier: prompts[tier] for tier in selected_tiers},
        extraction_config=bundle.rl["extraction"],
        token_counters=counters,
        models={tier: models[tier] for tier in selected_tiers},
        prices={tier: prices[tier] for tier in selected_tiers},
    )
    planned_extraction = dict(manifest.get("planned_extraction") or {})
    planned_extraction.update(selected_plan)
    truncation_plan_by_tier = dict(manifest.get("truncation_plan_by_tier") or {})
    for tier in selected_tiers:
        truncation_plan_by_tier[tier] = selected_truncation_plans.get(tier, {})
    manifest = update_experiment_manifest(
        manifest_path,
        planned_extraction=planned_extraction,
        planned_api_calls=sum(item["batch_count"] for item in planned_extraction.values()),
        last_selected_tiers=list(selected_tiers),
        extraction_segment_count=len(extraction_segments),
        truncation_plan_by_tier=truncation_plan_by_tier,
    )
    reliability = bundle.rl.get("api_reliability", {})
    client = OpenAICompatibleClient(
        timeout_seconds=int(reliability.get("timeout_seconds", 120)),
        max_retries=int(reliability.get("max_retries", 3)),
        retry_backoff_seconds=float(reliability.get("retry_backoff_seconds", 1.0)),
    )

    def complete(tier, prompt, max_new_tokens):
        response = client.complete(
            model_spec=models[tier],
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            json_mode=True,
        )
        return BatchCompletion(
            response.content,
            ProviderUsage(
                response.input_tokens,
                response.output_tokens,
                models[tier].model_name,
                response.usage_source,
                response.retry_count,
                response.latency_ms,
                response.provider_request_id,
                response.finish_reason,
            ),
            attempts=response.attempts or [],
        )

    store = FactQdrantStore.from_storage_config(
        storage_cfg,
        project_root=root,
        namespace=namespace,
    )
    try:
        update_experiment_manifest(manifest_path, status="active")
        generator = CandidateGenerator(
            store=store,
            encoder=encoder,
            models=models,
            prices=prices,
            token_counters=counters,
            completion=complete,
            prompts=prompts,
            prompt_versions=prompt_versions,
            extraction_config=bundle.rl["extraction"],
            output_root=output_root,
        )
        summary = generator.generate(
            extraction_segments,
            run_id,
            resume=resume,
            retry_terminal=args.retry_terminal,
            tiers=selected_tiers,
        )
        qdrant_counts = {
            tier: len(
                store.candidate_points(
                    tier,
                    dataset_name=first.dataset_name,
                    split=first.split,
                    sample_id=first.sample_id,
                    extraction_run_id=run_id,
                    with_vectors=False,
                )
            )
            for tier in models
        }
        qdrant_reconciled_by_tier = {
            tier: qdrant_counts[tier] == summary.fact_counts[tier] for tier in models
        }
        completed_tiers = []
        for tier in TIERS:
            plan = planned_extraction.get(tier)
            statuses = summary.batch_status_by_tier.get(tier, {})
            if (
                plan
                and statuses.get("committed", 0) == int(plan["batch_count"])
                and sum(statuses.values()) == int(plan["batch_count"])
                and qdrant_reconciled_by_tier[tier]
            ):
                completed_tiers.append(tier)
        selected_reconciled = all(
            qdrant_reconciled_by_tier[tier] for tier in selected_tiers
        )
        if set(completed_tiers) == set(TIERS):
            final_status = "complete"
        elif any(
            status != "committed"
            for tier_statuses in summary.batch_status_by_tier.values()
            for status in tier_statuses
        ):
            final_status = "incomplete"
        elif summary.status == "complete" and selected_reconciled:
            final_status = "partial"
        else:
            final_status = "incomplete"
        human = run_dir / "human_readable" / first.sample_id
        export_hashes = dict(manifest.get("export_sha256") or {})
        labels = {"small": "L", "medium": "M", "large": "H"}
        for tier in selected_tiers:
            label = labels[tier]
            export_path = export_memories(
                store,
                tier,
                dataset_name=first.dataset_name,
                split=first.split,
                sample_id=first.sample_id,
                extraction_run_id=run_id,
                output_path=human / f"{label}_memories.json",
            )
            export_hashes[label] = file_sha256(export_path)
        final_updates = {
            "status": final_status,
            "extraction_summary": asdict(summary),
            "qdrant_audit": {
                "counts_by_tier": qdrant_counts,
                "expected_by_tier": summary.fact_counts,
                "reconciled_by_tier": qdrant_reconciled_by_tier,
                "reconciled": all(qdrant_reconciled_by_tier.values()),
            },
            "export_sha256": export_hashes,
            "human_readable_export_dir": str(human),
            "qdrant_collection_namespace": namespace,
            "qdrant_collections": store.collections,
            "required_tiers": list(TIERS),
            "completed_tiers": completed_tiers,
            "last_selected_tiers": list(selected_tiers),
        }
        if final_status == "complete":
            final_updates["completed_at"] = datetime.now(timezone.utc).isoformat()
        update_experiment_manifest(
            manifest_path,
            **final_updates,
        )
        if args.campaign_id:
            refresh_campaign(bundle, args.campaign_id)
        result_payload = asdict(summary)
        result_payload.update(
            {
                "manifest_status": final_status,
                "completed_tiers": completed_tiers,
                "required_tiers": list(TIERS),
            }
        )
        print(json.dumps(result_payload, ensure_ascii=False, indent=2))
        if summary.status != "complete" or not selected_reconciled:
            raise SystemExit(2)
    except BaseException as exc:
        if not isinstance(exc, SystemExit):
            update_experiment_manifest(
                manifest_path,
                status="interrupted",
                last_error=f"{type(exc).__name__}: {exc}",
            )
        raise
    finally:
        store.close()


if __name__ == "__main__":
    try:
        main()
    except ProviderCircuitOpenError as exc:
        print(
            json.dumps(
                {
                    "status": "provider_circuit_open",
                    "tier": exc.tier,
                    "error": str(exc),
                },
                ensure_ascii=False,
            )
        )
        raise SystemExit(10) from exc
