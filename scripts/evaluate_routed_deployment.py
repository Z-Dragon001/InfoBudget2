"""Run a frozen router through real extraction, S assembly, and test QA evaluation."""

from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import torch

from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.splits import load_split_selection
from infobudget.rl_router.api import OpenAICompatibleClient, require_api_keys
from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.campaign import load_complete_campaign
from infobudget.rl_router.candidates import (
    CandidateGenerationSummary,
    CandidateGenerator,
    estimate_routed_plan,
    prepare_routed_extraction_segments,
)
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.costs import normalize_virtual_cost, replay_virtual_cost
from infobudget.rl_router.deployment import (
    build_question_outcomes,
    deployment_namespace,
    summarize_question_outcomes,
    summarize_deployment_costs,
    summarize_qa_usage,
    validate_route_decisions,
)
from infobudget.rl_router.embedding import LocalSentenceEncoder, LocalTokenizer
from infobudget.rl_router.evaluation import build_lightmem_evaluator
from infobudget.rl_router.experiment_identity import epoch_artifact_name
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.ledger import atomic_write_json, read_sqlite_ledger
from infobudget.rl_router.manifest import (
    create_experiment_manifest,
    file_sha256,
    memory_embedding_hash,
    resolve_collection_namespace,
    update_experiment_manifest,
)
from infobudget.rl_router.parsing import render_extraction_prompt
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.router import EmbeddingMLPRouter, SegmentFeatureBuilder
from infobudget.rl_router.schemas import BatchCompletion, ProviderUsage, TIERS
from infobudget.rl_router.training_io import discover_segment_files, load_replay_history
from infobudget.utils.logging import get_logger
from infobudget.utils.progress import StageProgress


logger = get_logger("routed_evaluation")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("split")
    parser.add_argument("--method", required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path)
    parser.add_argument("--campaign-id", required=True)
    run_group = parser.add_mutually_exclusive_group()
    run_group.add_argument("--deployment-run-id")
    run_group.add_argument("--resume", metavar="DEPLOYMENT_RUN_ID")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument(
        "--stage",
        choices=("all", "extract", "qa"),
        default="all",
        help="Run both stages, only routed memory extraction, or only QA/judging.",
    )
    args = parser.parse_args()
    if args.stage == "qa" and not args.resume:
        parser.error("--stage qa requires --resume DEPLOYMENT_RUN_ID")

    bundle = load_rl_bundle(args.config_dir)
    required_roles = {
        "all": (*TIERS, "qa_reader", "judge_llm"),
        "extract": TIERS,
        "qa": ("qa_reader", "judge_llm"),
    }[args.stage]
    require_api_keys(
        bundle.project.models,
        required_roles,
        operation=f"routed deployment stage={args.stage}",
    )
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    checkpoint_sha256 = file_sha256(checkpoint)
    training_manifest_path = (
        args.training_manifest.resolve()
        if args.training_manifest
        else checkpoint.parent.parent / "manifest.json"
    )
    training_manifest = _validate_training_manifest(
        training_manifest_path,
        checkpoint_sha256=checkpoint_sha256,
        dataset=args.dataset,
        split=args.split,
        method=args.method,
        fold=args.fold,
        split_manifest=args.split_manifest.resolve(),
        campaign_id=args.campaign_id,
    )
    campaign = load_complete_campaign(bundle, args.campaign_id)
    if (
        campaign["dataset_name"],
        campaign["split"],
        campaign["segmentation_method"],
    ) != (args.dataset, args.split, args.method):
        raise ValueError("deployment scope does not match the extraction campaign")

    available = set(
        discover_segment_files(
            bundle.project.root_dir, args.dataset, args.split, args.method
        )
    )
    selection = load_split_selection(
        args.split_manifest,
        dataset_name=args.dataset,
        source_split=args.split,
        fold=args.fold,
        available_sample_ids=available,
        source_processed_manifest_path=(
            bundle.project.root_dir
            / "datasets"
            / "processed"
            / args.dataset
            / args.split
            / "manifest.json"
        ),
        project_root=bundle.project.root_dir,
    )
    test_files = discover_segment_files(
        bundle.project.root_dir,
        args.dataset,
        args.split,
        args.method,
        list(selection.test_sample_ids),
    )
    segments_by_sample = {
        sample_id: load_segments(path) for sample_id, path in test_files.items()
    }
    _validate_segment_sets(segments_by_sample, args.dataset, args.split, args.method)
    examples = {
        item.sample_id: item
        for item in DatasetLoader(
            bundle.project.config.dataset, bundle.project.root_dir
        ).load(args.dataset, args.split, set(segments_by_sample))
    }
    if set(examples) != set(segments_by_sample):
        raise ValueError("processed test samples do not match the selected test fold")
    if any(not item.qa_pairs for item in examples.values()):
        raise ValueError("every test sample must contain at least one QA pair")
    excluded_categories = tuple(
        str(item)
        for item in (
            bundle.rl["evaluation"]
            .get("excluded_categories_by_dataset", {})
            .get(args.dataset, [])
        )
    )
    excluded_category_set = set(excluded_categories)
    evaluation_questions_by_sample = {
        sample_id: [
            pair
            for pair in example.qa_pairs
            if pair.category not in excluded_category_set
        ]
        for sample_id, example in examples.items()
    }
    if any(not values for values in evaluation_questions_by_sample.values()):
        raise ValueError(
            "category exclusions removed every evaluation question from a test sample"
        )
    evaluation_category_distribution = {}
    for pairs in evaluation_questions_by_sample.values():
        for pair in pairs:
            category = pair.category or "uncategorized"
            evaluation_category_distribution[category] = (
                evaluation_category_distribution.get(category, 0) + 1
            )

    device = _resolve_device(args.device)
    model, scaler, checkpoint_metadata = EmbeddingMLPRouter.load_checkpoint(
        checkpoint, device=device
    )
    router_encoder = _build_encoder(bundle, "router")
    memory_encoder = (
        router_encoder
        if bundle.embeddings["router"] == bundle.embeddings["memory"]
        else _build_encoder(bundle, "memory")
    )
    feature_builder = SegmentFeatureBuilder(router_encoder, scaler)
    routes_by_sample = {}
    actions_by_sample = {}
    for sample_id, segments in segments_by_sample.items():
        routes = model.route(feature_builder.build(segments), deterministic=True)
        actions = [route.tier for route in routes]
        validate_route_decisions(segments, actions)
        routes_by_sample[sample_id] = routes
        actions_by_sample[sample_id] = actions

    embedding_hash = memory_embedding_hash(bundle)
    source_namespace = resolve_collection_namespace(
        bundle.rl["storage"],
        model_family=bundle.rl["model_family"],
        dataset=args.dataset,
        split=args.split,
        segmentation_version=next(iter(segments_by_sample.values()))[0].segmentation_version,
        embedding_hash=embedding_hash,
    )
    deployment_run_id = (
        args.resume
        or args.deployment_run_id
        or datetime.now(timezone.utc).strftime("deploy_%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    namespace = deployment_namespace(
        source_namespace,
        protocol=selection.protocol,
        fold=selection.fold,
        deployment_run_id=deployment_run_id,
    )
    default_output = (
        bundle.project.root_dir
        / "outputs"
        / "rl_router"
        / "deployment_evaluation"
        / args.dataset
        / selection.protocol
        / f"fold_{selection.fold}"
        / args.method
        / epoch_artifact_name(int(training_manifest["epochs"]))
        / deployment_run_id
    )
    output_dir = (args.output_dir or default_output).resolve()
    manifest_path = output_dir / "manifest.json"
    resume = bool(args.resume)
    if resume:
        manifest = _validate_resume_manifest(
            manifest_path,
            deployment_run_id=deployment_run_id,
            checkpoint_sha256=checkpoint_sha256,
            split_manifest_sha256=selection.sha256,
            namespace=namespace,
        )
        if tuple(manifest.get("evaluation_excluded_categories") or ()) != excluded_categories:
            raise ValueError(
                "deployment resume uses different evaluation category exclusions"
            )
        if manifest.get("status") == "complete":
            raise ValueError("deployment evaluation is already complete")
    else:
        if manifest_path.exists():
            raise FileExistsError(
                f"deployment output already exists; use --resume {deployment_run_id}"
            )

    models = {tier: bundle.project.models[tier] for tier in TIERS}
    prices = {
        tier: bundle.project.prices[models[tier].model_name] for tier in TIERS
    }
    prompt = bundle.fact_extraction_prompt_path(args.dataset).read_text(encoding="utf-8")
    prompt_version = bundle.fact_extraction_prompt_version(args.dataset)
    prompts = {tier: prompt for tier in TIERS}
    prompt_versions = {tier: prompt_version for tier in TIERS}
    tokenizers = {
        tier: LocalTokenizer(bundle.project.root_dir / models[tier].tokenizer_local_path)
        for tier in TIERS
    }
    counters = {tier: tokenizers[tier].count for tier in TIERS}
    prepared_by_sample = {}
    truncation_by_sample = {}
    plans_by_sample = {}
    for sample_id, segments in segments_by_sample.items():
        prepared, truncation = prepare_routed_extraction_segments(
            segments=segments,
            actions=actions_by_sample[sample_id],
            prompts=prompts,
            extraction_config=bundle.rl["extraction"],
            token_counters=counters,
        )
        prepared_by_sample[sample_id] = prepared
        truncation_by_sample[sample_id] = truncation
        plans_by_sample[sample_id] = estimate_routed_plan(
            segments=prepared,
            actions=actions_by_sample[sample_id],
            prompts=prompts,
            extraction_config=bundle.rl["extraction"],
            token_counters=counters,
            models=models,
            prices=prices,
        )

    sample_run_ids = {
        sample_id: str(
            uuid.uuid5(uuid.NAMESPACE_URL, f"{deployment_run_id}:{sample_id}")
        )
        for sample_id in segments_by_sample
    }
    route_manifest = {
        sample_id: [
            {
                "segment_id": segment.segment_id,
                "tier": route.tier,
                "probability": route.probability,
                "probabilities": route.probabilities,
            }
            for segment, route in zip(segments_by_sample[sample_id], routes_by_sample[sample_id])
        ]
        for sample_id in segments_by_sample
    }
    if not resume:
        create_experiment_manifest(
            bundle,
            manifest_path,
            experiment_id=deployment_run_id,
            run_type="deployment_evaluation_v2",
            status="planned",
            dataset_name=args.dataset,
            split=args.split,
            segmentation_method=args.method,
            protocol=selection.protocol,
            fold=selection.fold,
            partition="test",
            test_sample_ids=list(selection.test_sample_ids),
            split_manifest={
                "path": str(selection.path),
                "sha256": selection.sha256,
            },
            training_manifest=str(training_manifest_path),
            training_experiment_id=training_manifest.get("experiment_id"),
            training_epochs_requested=training_manifest.get("epochs"),
            training_epochs_completed=training_manifest.get("epochs_completed"),
            checkpoint=str(checkpoint),
            checkpoint_sha256=checkpoint_sha256,
            checkpoint_metadata=checkpoint_metadata,
            source_campaign_id=args.campaign_id,
            source_campaign_scope_hash=campaign["campaign_scope_hash"],
            source_candidate_namespace=source_namespace,
            qdrant_collection_namespace=namespace,
            extraction_run_ids=sample_run_ids,
            route_decisions=route_manifest,
            planned_extraction=plans_by_sample,
            truncation_plan=truncation_by_sample,
            objective_cost_scope=bundle.rl["evaluation"]["objective_cost_scope"],
            requested_stage=args.stage,
            evaluation_excluded_categories=list(excluded_categories),
            evaluation_question_count=sum(
                len(items) for items in evaluation_questions_by_sample.values()
            ),
            evaluation_category_distribution=dict(
                sorted(evaluation_category_distribution.items())
            ),
            precomputed_embedding_hash=embedding_hash,
        )

    reliability = bundle.rl.get("api_reliability", {})
    client = OpenAICompatibleClient(
        timeout_seconds=int(reliability.get("timeout_seconds", 120)),
        max_retries=int(reliability.get("max_retries", 3)),
        retry_backoff_seconds=float(reliability.get("retry_backoff_seconds", 1.0)),
    )

    def complete(tier, rendered_prompt, max_new_tokens):
        response = client.complete(
            model_spec=models[tier],
            prompt=rendered_prompt,
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
        bundle.rl["storage"],
        project_root=bundle.project.root_dir,
        namespace=namespace,
    )
    run_extraction = args.stage in {"all", "extract"}
    run_qa = args.stage in {"all", "qa"}
    extraction_summaries: list[CandidateGenerationSummary] = []
    sample_results: dict[str, dict] = {}
    result_payload: dict
    extraction_progress = None
    qa_progress = None
    try:
        if run_extraction:
            update_experiment_manifest(manifest_path, status="extracting")
            extraction_progress = StageProgress(
                f"evaluation-memory extraction {args.dataset}/{args.method}",
                sum(
                    int(tier_plan["batch_count"])
                    for sample_plan in plans_by_sample.values()
                    for tier_plan in sample_plan.values()
                ),
                unit="batches",
            )

            def report_extraction_batch(event: dict) -> None:
                extraction_progress.update(
                    item=(
                        f"sample={event['sample_id']} "
                        f"tier={event['tier']} status={event['status']}"
                    ),
                    metrics={
                        "segments": event["segment_count"],
                        "facts": event["fact_count"],
                        "calls": event["logical_calls"],
                        "in_tok": event["input_tokens"],
                        "out_tok": event["output_tokens"],
                        "cost": event["known_cost"],
                    },
                )

            generator = CandidateGenerator(
                store=store,
                encoder=memory_encoder,
                models=models,
                prices=prices,
                token_counters=counters,
                completion=complete,
                prompts=prompts,
                prompt_versions=prompt_versions,
                extraction_config=bundle.rl["extraction"],
                output_root=output_dir / "extraction",
                ledger_filename="deployment_ledger.sqlite3",
                audit_context={
                    "model_family": bundle.rl["model_family"],
                    "campaign_id": args.campaign_id,
                    "campaign_scope_hash": campaign["campaign_scope_hash"],
                    "qdrant_namespace": namespace,
                    "qdrant_distance": bundle.rl["storage"].get(
                        "distance", "Cosine"
                    ),
                    "embedding_model_hash": embedding_hash,
                    "embedding_revision": bundle.embeddings["memory"].get(
                        "revision", ""
                    ),
                    "embedding_normalized": bool(
                        bundle.embeddings["memory"].get("normalize", False)
                    ),
                },
                progress_callback=report_extraction_batch,
            )
            for sample_id, segments in prepared_by_sample.items():
                summary = generator.generate_routed(
                    segments,
                    actions_by_sample[sample_id],
                    sample_run_ids[sample_id],
                    resume=resume,
                    route_scope={
                        "deployment_run_id": deployment_run_id,
                        "fold": selection.fold,
                        "checkpoint_sha256": checkpoint_sha256,
                    },
                )
                _reconcile_routed_sample(
                    output_dir=output_dir,
                    summary=summary,
                    segments=segments,
                    actions=actions_by_sample[sample_id],
                )
                if summary.status != "complete":
                    raise RuntimeError(
                        f"routed extraction is incomplete for {sample_id}"
                    )
                extraction_summaries.append(summary)

            question_count = sum(
                len(items) for items in evaluation_questions_by_sample.values()
            )
            extraction_costs = summarize_deployment_costs(
                extraction_summaries,
                question_count=question_count,
                currency=_single_currency(prices),
            )
            extraction_progress.close(
                metrics={
                    "samples": len(extraction_summaries),
                    "calls": extraction_costs["totals"]["logical_api_calls"],
                    "tokens": extraction_costs["totals"]["total_tokens"],
                    "cost": extraction_costs["totals"]["known_cost"],
                }
            )
            update_experiment_manifest(
                manifest_path,
                status="evaluating" if run_qa else "extraction_complete",
                extraction_completed_at=datetime.now(timezone.utc).isoformat(),
                extraction_costs=extraction_costs,
            )
        else:
            extraction_costs = dict(manifest.get("extraction_costs") or {})
            if not manifest.get("extraction_completed_at") or not extraction_costs:
                raise ValueError(
                    "QA stage requires a completed routed extraction stage in this deployment run"
                )
            update_experiment_manifest(manifest_path, status="evaluating")

        if run_qa:
            store.freeze_candidates()
            assembly_manager = AssemblyManager(
                store, output_dir / "routing" / "ledger.sqlite3"
            )
            evaluator = build_lightmem_evaluator(
                bundle,
                store=store,
                encoder=memory_encoder,
                client=client,
                ledger_path=output_dir / "qa" / "ledger.sqlite3",
            )
            qa_ledger_path = output_dir / "qa" / "ledger.sqlite3"
            virtual = _build_virtual_costs(
                bundle=bundle,
                campaign=campaign,
                segments_by_sample=segments_by_sample,
                actions_by_sample=actions_by_sample,
                prices=prices,
                tokenizers=tokenizers,
                prompt=prompt,
            )
            total_questions = sum(
                len(evaluation_questions_by_sample[sample_id])
                for sample_id in prepared_by_sample
            )
            qa_progress = StageProgress(
                f"QA + judge {args.dataset}/{args.method}",
                total_questions,
                unit="questions",
            )
            qa_correct = 0

            def report_qa(result) -> None:
                nonlocal qa_correct
                qa_correct += int(result.correct)
                qa_progress.update(
                    item=result.question_id,
                    metrics={
                        "correct": qa_correct,
                        "accuracy": qa_correct / max(1, qa_progress.completed + 1),
                        "retrieved": len(result.retrieved),
                        "tokens": (
                            result.reader_input_tokens
                            + result.reader_output_tokens
                            + result.judge_input_tokens
                            + result.judge_output_tokens
                        ),
                        "cost": (
                            result.reader_input_cost
                            + result.reader_output_cost
                            + result.judge_input_cost
                            + result.judge_output_cost
                        ),
                    },
                )

            for sample_id, segments in prepared_by_sample.items():
                questions = [
                    item.to_dict()
                    for item in evaluation_questions_by_sample[sample_id]
                ]
                result_path = output_dir / "samples" / sample_id / "result.json"
                if resume and result_path.is_file():
                    previous = json.loads(result_path.read_text(encoding="utf-8"))
                    stored = store.assembly_points(
                        dataset_name=args.dataset,
                        split=args.split,
                        sample_id=sample_id,
                        assembly_id=str(previous["assembly_id"]),
                        with_vectors=False,
                    )
                    if len(stored) != int(previous.get("s_fact_count", -1)):
                        raise RuntimeError(
                            f"completed sample assembly is missing or stale on resume: {sample_id}"
                        )
                    previous = _restore_sample_evaluation_metrics(
                        previous,
                        questions=questions,
                        qa_ledger_path=qa_ledger_path,
                    )
                    atomic_write_json(result_path, previous)
                    sample_results[sample_id] = previous
                    previous_questions = int(previous.get("question_count", 0))
                    qa_correct += int(previous.get("correct_count", 0))
                    qa_progress.update(
                        previous_questions,
                        item=f"sample={sample_id} resumed",
                        metrics={
                            "correct": qa_correct,
                            "accuracy": qa_correct / max(1, qa_progress.completed + previous_questions),
                        },
                    )
                    continue
                routes = routes_by_sample[sample_id]
                assembly = assembly_manager.create(
                    dataset_name=args.dataset,
                    split=args.split,
                    sample_id=sample_id,
                    segments=segments,
                    actions=actions_by_sample[sample_id],
                    probabilities=[item.probability for item in routes],
                    episode_id=f"deployment:{deployment_run_id}:{sample_id}",
                    policy_version=checkpoint_sha256[:16],
                    router_type="embedding_mlp",
                    candidate_extraction_run_id=sample_run_ids[sample_id],
                )
                if assembly.status != "ready":
                    raise RuntimeError(f"deployment assembly failed for {sample_id}")
                try:
                    score, evaluations = evaluator.evaluate_sample(
                        questions,
                        dataset_name=args.dataset,
                        split=args.split,
                        sample_id=sample_id,
                        assembly_id=assembly.assembly_id,
                        sample_metadata=examples[sample_id].metadata,
                        progress_callback=report_qa,
                    )
                except BaseException:
                    assembly_manager.cleanup(
                        assembly, dataset_name=args.dataset, split=args.split
                    )
                    raise
                question_outcomes = build_question_outcomes(questions, evaluations)
                sample_result = {
                    "sample_id": sample_id,
                    "status": "complete",
                    "assembly_id": assembly.assembly_id,
                    "s_fact_count": assembly.point_count,
                    "extraction_run_id": sample_run_ids[sample_id],
                    "segment_count": len(segments),
                    "tier_counts": {
                        tier: actions_by_sample[sample_id].count(tier) for tier in TIERS
                    },
                    "question_count": len(questions),
                    "correct_count": sum(item.correct for item in evaluations),
                    "qa_score": score,
                    "qa_usage": summarize_qa_usage(evaluations),
                    "question_outcomes": question_outcomes,
                    "question_metrics": summarize_question_outcomes(
                        question_outcomes
                    ),
                    "virtual_extraction": virtual[sample_id],
                }
                atomic_write_json(result_path, sample_result)
                sample_results[sample_id] = sample_result

            aggregate = _aggregate_results(sample_results, extraction_costs)
            aggregate.update(
                dataset_name=args.dataset,
                reader_model=bundle.project.models[
                    "qa_reader"
                ].effective_model_name,
                judge_model=bundle.project.models[
                    "judge_llm"
                ].effective_model_name,
                excluded_categories=list(excluded_categories),
            )
            qa_progress.close(
                metrics={
                    "correct": aggregate["correct_count"],
                    "accuracy": aggregate["qa_accuracy_micro"],
                    "tokens": aggregate["qa_usage"]["total_tokens"],
                    "cost": aggregate["qa_usage"]["total_cost"],
                }
            )
            atomic_write_json(output_dir / "aggregate.json", aggregate)
            update_experiment_manifest(
                manifest_path,
                status="complete",
                completed_at=datetime.now(timezone.utc).isoformat(),
                aggregate=aggregate,
                sample_result_files={
                    sample_id: str(output_dir / "samples" / sample_id / "result.json")
                    for sample_id in sample_results
                },
            )
            _log_evaluation_summary(aggregate, output_dir)
            result_payload = aggregate
        else:
            result_payload = {
                "status": "extraction_complete",
                "deployment_run_id": deployment_run_id,
                "output_dir": str(output_dir),
                "memory_extraction": extraction_costs,
            }
    except BaseException as exc:
        if extraction_progress is not None:
            extraction_progress.close(
                status="failed", metrics={"error": type(exc).__name__}
            )
        if qa_progress is not None:
            qa_progress.close(status="failed", metrics={"error": type(exc).__name__})
        update_experiment_manifest(
            manifest_path,
            status="interrupted",
            last_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        store.close()
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))


def _validate_training_manifest(
    path: Path,
    *,
    checkpoint_sha256: str,
    dataset: str,
    split: str,
    method: str,
    fold: int,
    split_manifest: Path,
    campaign_id: str,
) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"training manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError("deployment requires a completed training manifest")
    if (manifest.get("dataset_name"), manifest.get("split"), manifest.get("segmentation_method")) != (
        dataset,
        split,
        method,
    ):
        raise ValueError("checkpoint training scope does not match deployment scope")
    selected = str(manifest.get("selected_checkpoint_sha256") or "")
    if selected != checkpoint_sha256:
        raise ValueError("checkpoint is not the training manifest's selected checkpoint")
    declared_split = manifest.get("split_manifest") or {}
    if declared_split.get("sha256") != file_sha256(split_manifest) or declared_split.get("fold") != fold:
        raise ValueError("checkpoint was trained with a different split manifest or fold")
    if manifest.get("extraction_campaign_id") != campaign_id:
        raise ValueError("checkpoint was trained from a different extraction campaign")
    return manifest


def _validate_resume_manifest(path: Path, **expected) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"deployment manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    actual = {
        "deployment_run_id": manifest.get("experiment_id"),
        "checkpoint_sha256": manifest.get("checkpoint_sha256"),
        "split_manifest_sha256": (manifest.get("split_manifest") or {}).get("sha256"),
        "namespace": manifest.get("qdrant_collection_namespace"),
    }
    if actual != expected:
        raise ValueError(f"deployment resume scope mismatch: expected={expected}, actual={actual}")
    return manifest


def _reconcile_routed_sample(*, output_dir, summary, segments, actions) -> None:
    first = segments[0]
    ledger = (
        output_dir
        / "extraction"
        / first.dataset_name
        / first.split
        / first.segmentation_method
        / "samples"
        / first.sample_id
        / "extraction"
        / "deployment_ledger.sqlite3"
    )
    rows = [
        row
        for row in read_sqlite_ledger(ledger, "segment_costs")
        if row.get("extraction_run_id") == summary.extraction_run_id
    ]
    expected = validate_route_decisions(segments, actions)
    actual = {str(row["segment_id"]): str(row["tier"]) for row in rows}
    if len(rows) != len(expected) or actual != expected:
        raise RuntimeError(
            f"routed extraction reconciliation failed for {first.sample_id}: "
            f"expected={expected}, actual={actual}"
        )


def _build_virtual_costs(*, bundle, campaign, segments_by_sample, actions_by_sample, prices, tokenizers, prompt):
    shared_prompt_tokens = {
        tier: tokenizers[tier].count(render_extraction_prompt(prompt, tier, []))
        for tier in TIERS
    }
    sample_root = (
        bundle.project.root_dir
        / "outputs"
        / "rl_router"
        / campaign["dataset_name"]
        / campaign["split"]
        / campaign["segmentation_method"]
        / "samples"
    )
    result = {}
    for sample_id, segments in segments_by_sample.items():
        run_id = campaign["expected_runs"][sample_id]
        _, history = load_replay_history(
            sample_root / sample_id / "extraction" / "candidate_ledger.sqlite3",
            segments,
            run_id,
        )
        calculate = partial(
            replay_virtual_cost,
            segments,
            historical=history,
            buffer_config=bundle.rl["extraction"]["buffers"],
            prices=prices,
            shared_prompt_tokens=shared_prompt_tokens,
        )
        routed = calculate(actions_by_sample[sample_id])
        all_small = calculate(["small"] * len(segments)).total_cost
        all_large = calculate(["large"] * len(segments)).total_cost
        result[sample_id] = {
            **asdict(routed),
            "total_tokens": routed.total_tokens,
            "total_cost": routed.total_cost,
            "normalized_cost": normalize_virtual_cost(
                routed.total_cost, all_small, all_large
            ),
            "all_small_cost": all_small,
            "all_large_cost": all_large,
        }
    return result


def _restore_sample_evaluation_metrics(
    previous: dict,
    *,
    questions: list[dict],
    qa_ledger_path: Path,
) -> dict:
    """Backfill classification metrics from the authoritative per-question ledger."""
    if not qa_ledger_path.is_file():
        raise FileNotFoundError(
            "cannot resume classified QA metrics without the evaluation ledger: "
            f"{qa_ledger_path}"
        )
    allowed_ids = {str(item["question_id"]) for item in questions}
    rows = [
        row
        for row in read_sqlite_ledger(qa_ledger_path, "evaluations")
        if row.get("sample_id") == previous.get("sample_id")
        and str(row.get("question_id")) in allowed_ids
    ]
    outcomes = build_question_outcomes(questions, rows)
    correct = sum(bool(item["correct"]) for item in outcomes)
    restored = dict(previous)
    restored.update(
        question_count=len(outcomes),
        correct_count=correct,
        qa_score=correct / len(outcomes) if outcomes else 0.0,
        qa_usage=summarize_qa_usage(rows),
        question_outcomes=outcomes,
        question_metrics=summarize_question_outcomes(outcomes),
    )
    return restored


def _aggregate_results(sample_results: dict, extraction_costs: dict) -> dict:
    values = list(sample_results.values())
    questions = sum(int(item["question_count"]) for item in values)
    correct = sum(int(item["correct_count"]) for item in values)
    outcomes = [
        outcome
        for item in values
        for outcome in item.get("question_outcomes", [])
    ]
    if len(outcomes) != questions:
        raise RuntimeError(
            "sample results are missing classified question outcomes: "
            f"expected {questions}, got {len(outcomes)}"
        )
    question_metrics = summarize_question_outcomes(outcomes)
    qa_usage = _aggregate_sample_qa_usage(values)
    virtual_cost = sum(float(item["virtual_extraction"]["total_cost"]) for item in values)
    virtual_calls = sum(
        sum(int(count) for count in item["virtual_extraction"]["batch_count_by_tier"].values())
        for item in values
    )
    actual_cost = float(extraction_costs["totals"]["known_cost"])
    actual_calls = int(extraction_costs["totals"]["logical_api_calls"])
    aggregate_metrics = {
        "overall": {
            "judge_correct": question_metrics["overall"]["judge_correct"]
        }
    }
    aggregate_metrics.update(
        {
            category: {"judge_correct": metrics["judge_correct"]}
            for category, metrics in question_metrics["by_category"].items()
        }
    )
    category_distribution = {
        category.removeprefix("category_"): int(metrics["count"])
        for category, metrics in question_metrics["by_category"].items()
    }
    token_statistics = {
        "scope": "qa_reader_plus_judge",
        "total_api_calls": qa_usage["logical_api_calls"],
        "total_prompt_tokens": qa_usage["input_tokens"],
        "total_completion_tokens": qa_usage["output_tokens"],
        "total_tokens": qa_usage["total_tokens"],
        "avg_prompt_tokens_per_call": (
            qa_usage["input_tokens"] / qa_usage["logical_api_calls"]
            if qa_usage["logical_api_calls"]
            else 0.0
        ),
        "avg_completion_tokens_per_call": (
            qa_usage["output_tokens"] / qa_usage["logical_api_calls"]
            if qa_usage["logical_api_calls"]
            else 0.0
        ),
        "avg_total_tokens_per_call": (
            qa_usage["total_tokens"] / qa_usage["logical_api_calls"]
            if qa_usage["logical_api_calls"]
            else 0.0
        ),
        "reader": qa_usage["reader"],
        "judge": qa_usage["judge"],
    }
    return {
        "schema_version": "deployment_evaluation_aggregate_v2",
        "sample_count": len(values),
        "question_count": questions,
        "correct_count": correct,
        "qa_accuracy_micro": correct / questions if questions else 0.0,
        "qa_accuracy_macro": (
            sum(float(item["qa_score"]) for item in values) / len(values) if values else 0.0
        ),
        "question_outcomes": outcomes,
        "question_metrics": question_metrics,
        "category_distribution": category_distribution,
        "aggregate_metrics": aggregate_metrics,
        "token_statistics": token_statistics,
        "memory_extraction": extraction_costs,
        "qa_usage": qa_usage,
        "virtual_extraction_cost": virtual_cost,
        "virtual_extraction_calls": virtual_calls,
        "actual_minus_virtual_extraction_cost": actual_cost - virtual_cost,
        "actual_minus_virtual_extraction_calls": actual_calls - virtual_calls,
        "end_to_end_known_cost": actual_cost + qa_usage["total_cost"],
    }


def _aggregate_sample_qa_usage(values: list[dict]) -> dict:
    result: dict = {"question_count": sum(int(item["question_count"]) for item in values)}
    summed_fields = (
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "input_cost",
        "output_cost",
        "total_cost",
        "logical_api_calls",
        "reported_retry_count",
        "transport_attempts",
    )
    for role in ("reader", "judge"):
        result[role] = {
            field: sum(item["qa_usage"][role][field] for item in values)
            for field in summed_fields
        }
    result.update(
        input_tokens=result["reader"]["input_tokens"] + result["judge"]["input_tokens"],
        output_tokens=result["reader"]["output_tokens"] + result["judge"]["output_tokens"],
        total_tokens=result["reader"]["total_tokens"] + result["judge"]["total_tokens"],
        input_cost=result["reader"]["input_cost"] + result["judge"]["input_cost"],
        output_cost=result["reader"]["output_cost"] + result["judge"]["output_cost"],
        total_cost=result["reader"]["total_cost"] + result["judge"]["total_cost"],
        logical_api_calls=(
            result["reader"]["logical_api_calls"]
            + result["judge"]["logical_api_calls"]
        ),
        transport_attempts=(
            result["reader"]["transport_attempts"]
            + result["judge"]["transport_attempts"]
        ),
        reader_cost=result["reader"]["total_cost"],
        judge_cost=result["judge"]["total_cost"],
    )
    return result


def _log_evaluation_summary(aggregate: dict, output_dir: Path) -> None:
    separator = "=" * 80
    logger.info(separator)
    logger.info("Evaluation Complete")
    logger.info(separator)
    logger.info("Total samples:    %s", aggregate["sample_count"])
    logger.info("Total questions:  %s", aggregate["question_count"])
    logger.info("Reader model:     %s", aggregate["reader_model"])
    logger.info("Judge model:      %s", aggregate["judge_model"])
    if aggregate.get("excluded_categories"):
        logger.info(
            "Excluded categories: %s",
            ", ".join(aggregate["excluded_categories"]),
        )

    logger.info("Category Distribution:")
    category_metrics = aggregate["question_metrics"]["by_category"]
    for metrics in category_metrics.values():
        logger.info(
            "  %s: %s questions (%.1f%%)",
            metrics["label"],
            metrics["count"],
            100.0 * metrics["fraction"],
        )

    logger.info("Aggregate Metrics:")
    groups = [("Overall", aggregate["question_metrics"]["overall"])]
    groups.extend(
        (metrics["label"], metrics) for metrics in category_metrics.values()
    )
    for label, metrics in groups:
        judge = metrics["judge_correct"]
        logger.info("  %s judge_correct:", label)
        logger.info("    mean: %.4f", judge["mean"])
        logger.info("    std:  %.4f", judge["std"])
        logger.info("    count: %d", judge["count"])

    question_types = aggregate["question_metrics"]["by_question_type"]
    if set(question_types) != {"uncategorized"}:
        logger.info("Question Type Metrics:")
        for metrics in question_types.values():
            judge = metrics["judge_correct"]
            logger.info(
                "  %s: mean=%.4f std=%.4f count=%d (%.1f%%)",
                metrics["label"],
                judge["mean"],
                judge["std"],
                judge["count"],
                100.0 * metrics["fraction"],
            )

    token_stats = aggregate["token_statistics"]
    logger.info("Token Statistics (Reader + Judge):")
    logger.info("  Total API calls:         %s", token_stats["total_api_calls"])
    logger.info("  Total prompt tokens:     %s", f"{token_stats['total_prompt_tokens']:,}")
    logger.info(
        "  Total completion tokens: %s",
        f"{token_stats['total_completion_tokens']:,}",
    )
    logger.info("  Total tokens:            %s", f"{token_stats['total_tokens']:,}")
    logger.info(
        "  Avg prompt/call:         %.2f",
        token_stats["avg_prompt_tokens_per_call"],
    )
    logger.info(
        "  Avg completion/call:     %.2f",
        token_stats["avg_completion_tokens_per_call"],
    )
    logger.info(
        "  Reader: calls=%d tokens=%s cost=%.6f",
        token_stats["reader"]["logical_api_calls"],
        f"{token_stats['reader']['total_tokens']:,}",
        token_stats["reader"]["total_cost"],
    )
    logger.info(
        "  Judge:  calls=%d tokens=%s cost=%.6f",
        token_stats["judge"]["logical_api_calls"],
        f"{token_stats['judge']['total_tokens']:,}",
        token_stats["judge"]["total_cost"],
    )
    logger.info("Results saved to: %s", output_dir / "aggregate.json")
    logger.info(separator)


def _single_currency(prices: dict) -> str:
    currencies = {str(item.currency) for item in prices.values()}
    if len(currencies) != 1:
        raise ValueError(f"deployment extraction mixes currencies: {sorted(currencies)}")
    return currencies.pop()


def _build_encoder(bundle, role: str) -> LocalSentenceEncoder:
    config = bundle.embeddings[role]
    return LocalSentenceEncoder(
        model_name=config["model_name"],
        local_path=bundle.project.root_dir / config["local_path"],
        dimension=config["dimension"],
        normalize=config["normalize"],
        max_length=config.get("max_length"),
        long_text_strategy=config.get("long_text_strategy", "truncate"),
    )


def _resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return value


def _validate_segment_sets(segments_by_sample, dataset, split, method) -> None:
    versions = set()
    for sample_id, segments in segments_by_sample.items():
        for segment in segments:
            if (
                segment.dataset_name,
                segment.split,
                segment.sample_id,
                segment.segmentation_method,
            ) != (dataset, split, sample_id, method):
                raise ValueError(f"segment scope mismatch in test sample {sample_id}")
            versions.add(segment.segmentation_version)
    if len(versions) != 1:
        raise ValueError(f"test samples mix segmentation versions: {sorted(versions)}")


if __name__ == "__main__":
    main()
