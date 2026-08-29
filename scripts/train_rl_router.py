"""Train the budget-constrained Embedding+MLP router over frozen L/M/H candidates."""

from __future__ import annotations

import argparse
import json
import random
import uuid
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch

from infobudget.datasets.loader import DatasetLoader
from infobudget.datasets.splits import load_split_selection
from infobudget.rl_router.assembly import AssemblyManager
from infobudget.rl_router.campaign import load_complete_campaign
from infobudget.rl_router.config import load_rl_bundle
from infobudget.rl_router.costs import normalize_virtual_cost, replay_virtual_cost
from infobudget.rl_router.embedding import LocalSentenceEncoder, LocalTokenizer
from infobudget.rl_router.evaluation import build_lightmem_evaluator
from infobudget.rl_router.experiment import RLExperimentTrainer
from infobudget.rl_router.experiment_identity import epoch_artifact_name
from infobudget.rl_router.io import load_segments
from infobudget.rl_router.ledger import SqliteLedger, read_sqlite_ledger
from infobudget.rl_router.manifest import (
    create_experiment_manifest,
    file_sha256,
    memory_embedding_hash,
    resolve_collection_namespace,
    update_experiment_manifest,
)
from infobudget.rl_router.parsing import render_extraction_prompt
from infobudget.rl_router.qdrant_store import FactQdrantStore
from infobudget.rl_router.reconciliation import reconcile_extraction_run
from infobudget.rl_router.router import EmbeddingMLPRouter, SegmentFeatureBuilder
from infobudget.rl_router.training import ConstrainedActorCriticTrainer
from infobudget.rl_router.training_io import discover_segment_files, load_replay_history
from infobudget.utils.progress import StageProgress


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["locomo", "longmemeval"])
    parser.add_argument("split")
    parser.add_argument("--method", required=True)
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--extraction-run-id", help="Only valid when exactly one sample is selected.")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-sample", type=int, default=1)
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--policy-version")
    parser.add_argument("--run-id")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoints/training_state.pt in --output-dir.",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--config-dir", type=Path, default=Path("configs"))
    parser.add_argument("--campaign-id", required=True)
    args = parser.parse_args()
    if args.epochs <= 0 or args.steps_per_sample <= 0 or args.early_stopping_patience <= 0:
        parser.error("--epochs, --steps-per-sample, and --early-stopping-patience must be positive")
    if (args.split_manifest is None) != (args.fold is None):
        parser.error("--split-manifest and --fold must be provided together")
    if args.split_manifest and args.sample_ids:
        parser.error("--sample-id cannot be combined with --split-manifest")
    if args.split_manifest is None:
        parser.error("router training requires --split-manifest and --fold")
    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir")
    if args.resume and args.run_id:
        parser.error("--run-id cannot be combined with --resume")

    bundle = load_rl_bundle(args.config_dir)
    _require_training_api_keys(bundle)
    FactQdrantStore.probe_storage_config(bundle.rl["storage"])
    campaign = load_complete_campaign(bundle, args.campaign_id)
    if (
        campaign["dataset_name"],
        campaign["split"],
        campaign["segmentation_method"],
    ) != (args.dataset, args.split, args.method):
        raise ValueError("training scope does not match the extraction campaign")
    selection = None
    selected_sample_ids = args.sample_ids
    if args.split_manifest:
        available = set(
            discover_segment_files(
                bundle.project.root_dir,
                args.dataset,
                args.split,
                args.method,
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
        selected_sample_ids = list(selection.train_sample_ids)
    segment_files = discover_segment_files(
        bundle.project.root_dir,
        args.dataset,
        args.split,
        args.method,
        selected_sample_ids,
    )
    validation_segment_files = (
        discover_segment_files(
            bundle.project.root_dir,
            args.dataset,
            args.split,
            args.method,
            list(selection.validation_sample_ids),
        )
        if selection is not None and selection.validation_sample_ids
        else {}
    )
    if args.extraction_run_id and len(segment_files) != 1:
        parser.error("--extraction-run-id requires exactly one selected sample")
    segments_by_sample = {sample_id: load_segments(path) for sample_id, path in segment_files.items()}
    validation_segments_by_sample = {
        sample_id: load_segments(path) for sample_id, path in validation_segment_files.items()
    }
    all_selected_segments = {**segments_by_sample, **validation_segments_by_sample}
    _validate_segment_sets(all_selected_segments, args.dataset, args.split, args.method)

    examples = {
        example.sample_id: example
        for example in DatasetLoader(bundle.project.config.dataset, bundle.project.root_dir).load(
            args.dataset,
            args.split,
            set(all_selected_segments),
        )
    }
    missing_examples = sorted(set(all_selected_segments) - examples.keys())
    if missing_examples:
        raise ValueError(f"processed samples are missing: {missing_examples}")
    if any(not examples[sample_id].qa_pairs for sample_id in all_selected_segments):
        raise ValueError("every selected train/validation sample must contain at least one QA pair")

    sample_root = bundle.project.root_dir / "outputs" / "rl_router" / args.dataset / args.split / args.method / "samples"
    histories = {}
    extraction_runs = {}
    for sample_id, segments in all_selected_segments.items():
        expected_run_id = campaign["expected_runs"].get(sample_id)
        if not expected_run_id:
            raise ValueError(f"sample {sample_id} is absent from extraction campaign")
        if args.extraction_run_id and args.extraction_run_id != expected_run_id:
            raise ValueError("--extraction-run-id does not match the extraction campaign")
        run_id, history = load_replay_history(
            sample_root / sample_id / "extraction" / "candidate_ledger.sqlite3",
            segments,
            expected_run_id,
        )
        extraction_runs[sample_id] = run_id
        histories[sample_id] = history

    reconciliation_first = next(iter(all_selected_segments.values()))[0]
    reconciliation_embedding_hash = memory_embedding_hash(bundle)
    reconciliation_namespace = resolve_collection_namespace(
        bundle.rl["storage"],
        model_family=bundle.rl["model_family"],
        dataset=args.dataset,
        split=args.split,
        segmentation_version=reconciliation_first.segmentation_version,
        embedding_hash=reconciliation_embedding_hash,
    )
    reconciliation_store = FactQdrantStore.from_storage_config(
        bundle.rl["storage"],
        project_root=bundle.project.root_dir,
        namespace=reconciliation_namespace,
        read_only=True,
    )
    reconciliation_results = {}
    try:
        for expected_run_id in sorted(set(extraction_runs.values())):
            reconciliation_results[expected_run_id] = reconcile_extraction_run(
                bundle.project.root_dir,
                expected_run_id,
                reconciliation_store,
                raise_on_error=True,
            )
    finally:
        reconciliation_store.close()

    segmentation_version = next(iter(all_selected_segments.values()))[0].segmentation_version
    proposed_run_id = args.run_id or (
        datetime.now(timezone.utc).strftime(f"train_e{args.epochs}_%Y%m%dT%H%M%SZ_")
        + uuid.uuid4().hex[:8]
    )
    epoch_scope = epoch_artifact_name(args.epochs)
    if selection is None:
        default_output = bundle.project.root_dir / "outputs" / "rl_router" / "training" / args.dataset / args.split / args.method / epoch_scope / proposed_run_id
    else:
        default_output = bundle.project.root_dir / "outputs" / "rl_router" / "training" / args.dataset / selection.protocol / f"fold_{selection.fold}" / args.method / epoch_scope / proposed_run_id
    output_dir = (args.output_dir or default_output).resolve()
    existing_manifest_path = output_dir / "manifest.json"
    if args.resume:
        if not existing_manifest_path.is_file():
            raise FileNotFoundError(f"training resume manifest is missing: {existing_manifest_path}")
        existing_manifest = json.loads(existing_manifest_path.read_text(encoding="utf-8"))
        run_id = str(existing_manifest.get("experiment_id") or "")
        if not run_id:
            raise ValueError("training resume manifest has no experiment_id")
    else:
        run_id = proposed_run_id
    if not args.resume and (
        (output_dir / "training_ledger.sqlite3").exists()
        or (output_dir / "checkpoints").exists()
        or existing_manifest_path.exists()
    ):
        raise FileExistsError(f"training output already exists; choose a new --output-dir: {output_dir}")

    router_encoder = _build_encoder(bundle, "router")
    memory_encoder = router_encoder if bundle.embeddings["router"] == bundle.embeddings["memory"] else _build_encoder(bundle, "memory")
    all_segments = [segment for sample_id in segments_by_sample for segment in segments_by_sample[sample_id]]
    feature_builder = SegmentFeatureBuilder(router_encoder)
    scaler = feature_builder.fit(all_segments)
    features_by_sample = {
        sample_id: feature_builder.build(segments)
        for sample_id, segments in segments_by_sample.items()
    }
    validation_features_by_sample = {
        sample_id: feature_builder.build(segments)
        for sample_id, segments in validation_segments_by_sample.items()
    }

    router_cfg = bundle.rl["router"]
    seed = int(bundle.rl["seed"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = EmbeddingMLPRouter(
        router_encoder.dimension + SegmentFeatureBuilder.numeric_dimension,
        [int(value) for value in router_cfg["hidden_dimensions"]],
        float(router_cfg["dropout"]),
    )
    device = _resolve_device(args.device)
    model.to(device)
    optimizer = ConstrainedActorCriticTrainer(
        model,
        budget=float(router_cfg["budget"]),
        learning_rate=float(router_cfg["learning_rate"]),
        lambda_learning_rate=float(router_cfg["lambda_learning_rate"]),
        value_loss_coefficient=float(router_cfg.get("value_loss_coefficient", 0.5)),
        entropy_coefficient=float(router_cfg.get("entropy_coefficient", 0.01)),
        max_gradient_norm=float(router_cfg.get("max_gradient_norm", 1.0)),
        seed=seed,
    )

    first = all_segments[0]
    storage = bundle.rl["storage"]
    embedding_hash = memory_embedding_hash(bundle)
    namespace = resolve_collection_namespace(
        storage,
        model_family=bundle.rl["model_family"],
        dataset=args.dataset,
        split=args.split,
        segmentation_version=first.segmentation_version,
        embedding_hash=embedding_hash,
    )
    store = FactQdrantStore.from_storage_config(
        storage,
        project_root=bundle.project.root_dir,
        namespace=namespace,
    )
    store.freeze_candidates()
    assembly_manager = AssemblyManager(store, output_dir / "routing" / "ledger.sqlite3")
    evaluator = build_lightmem_evaluator(
        bundle,
        store=store,
        encoder=memory_encoder,
        ledger_path=output_dir / "qa" / "ledger.sqlite3",
    )
    validation_evaluator = (
        build_lightmem_evaluator(
            bundle,
            store=store,
            encoder=memory_encoder,
            ledger_path=output_dir / "validation" / "evaluations.sqlite3",
        )
        if validation_segments_by_sample
        else None
    )
    experiment = RLExperimentTrainer(
        model=model,
        scaler=scaler,
        trainer=optimizer,
        assembly_manager=assembly_manager,
        evaluator=evaluator,
        virtual_cost=None,
        output_dir=output_dir,
    )
    prices = {
        tier: bundle.project.prices[bundle.project.models[tier].model_name]
        for tier in ("small", "medium", "large")
    }
    prompt = bundle.fact_extraction_prompt_path(args.dataset).read_text(encoding="utf-8")
    tokenizers = {
        tier: LocalTokenizer(bundle.project.root_dir / bundle.project.models[tier].tokenizer_local_path)
        for tier in ("small", "medium", "large")
    }
    shared_prompt_tokens = {
        tier: tokenizers[tier].count(render_extraction_prompt(prompt, tier, []))
        for tier in tokenizers
    }
    virtual_costs = {
        sample_id: partial(
            replay_virtual_cost,
            segments,
            historical=histories[sample_id],
            buffer_config=bundle.rl["extraction"]["buffers"],
            prices=prices,
            shared_prompt_tokens=shared_prompt_tokens,
        )
        for sample_id, segments in all_selected_segments.items()
    }
    baselines = {}
    normalizers = {}
    for sample_id, segments in all_selected_segments.items():
        all_small = virtual_costs[sample_id](["small"] * len(segments)).total_cost
        all_large = virtual_costs[sample_id](["large"] * len(segments)).total_cost
        # Validate the normalization before any paid QA calls are made.
        normalize_virtual_cost(all_small, all_small, all_large)
        baselines[sample_id] = {"all_small": all_small, "all_large": all_large}
        normalizers[sample_id] = partial(normalize_virtual_cost, all_small=all_small, all_large=all_large)

    policy_version = args.policy_version or run_id
    manifest_path = output_dir / "manifest.json"
    manifest_scope = {
        "dataset_name": args.dataset,
        "split": args.split,
        "segmentation_method": args.method,
        "segmentation_version": segmentation_version,
        "sample_ids": list(segments_by_sample),
        "validation_sample_ids": list(validation_segments_by_sample),
        "extraction_campaign_id": args.campaign_id,
        "extraction_campaign_scope_hash": campaign["campaign_scope_hash"],
        "epochs": args.epochs,
        "steps_per_sample": args.steps_per_sample,
        "early_stopping_patience": args.early_stopping_patience,
        "device": device,
        "split_manifest": (
            {
                "path": str(selection.path),
                "sha256": selection.sha256,
                "protocol": selection.protocol,
                "fold": selection.fold,
                "train_sample_ids": list(selection.train_sample_ids),
                "validation_sample_ids": list(selection.validation_sample_ids),
                "test_sample_ids": list(selection.test_sample_ids),
            }
            if selection is not None
            else None
        ),
    }
    if args.resume:
        _validate_training_resume_manifest(
            manifest_path,
            run_id=run_id,
            scope=manifest_scope,
        )
    else:
        create_experiment_manifest(
            bundle,
            manifest_path,
            experiment_id=run_id,
            run_type="router_training",
            status="planned",
            extraction_runs=extraction_runs,
            extraction_reconciliation=reconciliation_results,
            cost_baselines=baselines,
            **manifest_scope,
            precomputed_embedding_hash=embedding_hash,
        )
    planned_episodes = args.epochs * args.steps_per_sample * len(segments_by_sample)
    planned_questions = args.epochs * args.steps_per_sample * sum(
        len(examples[sample_id].qa_pairs) for sample_id in segments_by_sample
    )
    planned_validation_questions = args.epochs * sum(
        len(examples[sample_id].qa_pairs) for sample_id in validation_segments_by_sample
    )
    print(
        json.dumps(
            {
                "status": "starting",
                "samples": len(segments_by_sample),
                "segments": len(all_segments),
                "episodes": planned_episodes,
                "qa_reader_calls": planned_questions,
                "judge_calls": planned_questions,
                "validation_qa_reader_calls": planned_validation_questions,
                "validation_judge_calls": planned_validation_questions,
                "device": device,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )
    validation_ledger = SqliteLedger(
        output_dir / "validation" / "ledger.sqlite3",
        "epochs",
        ("epoch", "sample_id"),
        legacy_jsonl_path=output_dir / "validation" / "epochs.jsonl",
    )
    best_validation: tuple[float, float] | None = None
    epochs_without_improvement = 0
    completed_epochs = 0
    stopped_early = False
    start_epoch = 0
    start_sample_index = 0
    state_path = output_dir / "checkpoints" / "training_state.pt"
    if args.resume:
        loop_state = _load_training_state(
            state_path,
            model=model,
            optimizer=optimizer,
            experiment=experiment,
            expected_run_id=run_id,
        )
        start_epoch = int(loop_state["epoch_index"])
        start_sample_index = int(loop_state["next_sample_index"])
        completed_epochs = int(loop_state["completed_epochs"])
        epochs_without_improvement = int(loop_state["epochs_without_improvement"])
        stopped_early = bool(loop_state.get("stopped_early", False))
        raw_best = loop_state.get("best_validation")
        best_validation = tuple(raw_best) if raw_best is not None else None
        if start_epoch > args.epochs:
            raise ValueError("training state is beyond the configured epoch count")
        if stopped_early:
            start_epoch = args.epochs
    train_items = list(segments_by_sample.items())
    if not 0 <= start_sample_index <= len(train_items):
        raise ValueError("training state next_sample_index is outside the training partition")
    training_progress = StageProgress(
        f"router training {args.dataset}/{args.method}/epoch_{args.epochs}",
        planned_episodes,
        unit="episodes",
        initial=min(
            planned_episodes,
            (start_epoch * len(train_items) + start_sample_index)
            * args.steps_per_sample,
        ),
    )
    latest = None
    try:
        update_experiment_manifest(manifest_path, status="active")
        if not state_path.exists():
            _save_training_state(
                state_path,
                run_id=run_id,
                model=model,
                optimizer=optimizer,
                experiment=experiment,
                epoch_index=start_epoch,
                next_sample_index=start_sample_index,
                completed_epochs=completed_epochs,
                epochs_without_improvement=epochs_without_improvement,
                best_validation=best_validation,
                stopped_early=stopped_early,
            )
        for epoch in range(start_epoch, args.epochs):
            sample_offset = start_sample_index if epoch == start_epoch else 0
            for sample_index, (sample_id, segments) in enumerate(
                train_items[sample_offset:], start=sample_offset
            ):
                history = experiment.train_sample(
                    features=features_by_sample[sample_id],
                    segments=segments,
                    questions=[pair.to_dict() for pair in examples[sample_id].qa_pairs],
                    steps=args.steps_per_sample,
                    policy_version=policy_version,
                    sample_metadata=examples[sample_id].metadata,
                    virtual_cost=virtual_costs[sample_id],
                    cost_normalizer=normalizers[sample_id],
                    candidate_extraction_run_id=extraction_runs[sample_id],
                    save_final=False,
                    track_best=not validation_segments_by_sample,
                )
                latest = history[-1]
                training_progress.update(
                    args.steps_per_sample,
                    item=f"epoch={epoch + 1} sample={sample_id}",
                    metrics={
                        "qa": latest.qa_score,
                        "norm_cost": latest.virtual_cost,
                        "reward": latest.reward,
                        "lambda": latest.lagrange_multiplier,
                    },
                )
                _save_training_state(
                    state_path,
                    run_id=run_id,
                    model=model,
                    optimizer=optimizer,
                    experiment=experiment,
                    epoch_index=epoch,
                    next_sample_index=sample_index + 1,
                    completed_epochs=completed_epochs,
                    epochs_without_improvement=epochs_without_improvement,
                    best_validation=best_validation,
                    stopped_early=stopped_early,
                )
            completed_epochs = epoch + 1
            if validation_segments_by_sample:
                validation_score, validation_cost = _evaluate_validation_epoch(
                    epoch=completed_epochs,
                    model=model,
                    features_by_sample=validation_features_by_sample,
                    segments_by_sample=validation_segments_by_sample,
                    examples=examples,
                    assembly_manager=assembly_manager,
                    evaluator=validation_evaluator,
                    virtual_costs=virtual_costs,
                    normalizers=normalizers,
                    extraction_runs=extraction_runs,
                    policy_version=policy_version,
                    ledger=validation_ledger,
                )
                within_budget = validation_cost <= float(router_cfg["budget"])
                metric = (validation_score, -validation_cost)
                improved = within_budget and (best_validation is None or metric > best_validation)
                if improved:
                    best_validation = metric
                    epochs_without_improvement = 0
                    model.save_checkpoint(
                        output_dir / "checkpoints" / "best.pt",
                        scaler,
                        {
                            "selected_by": "validation",
                            "epoch": completed_epochs,
                            "validation_qa_score": validation_score,
                            "validation_normalized_cost": validation_cost,
                        },
                    )
                else:
                    epochs_without_improvement += 1
                print(
                    json.dumps(
                        {
                            "epoch": completed_epochs,
                            "partition": "validation",
                            "qa_score": validation_score,
                            "normalized_cost": validation_cost,
                            "within_budget": within_budget,
                            "best": improved,
                        },
                        ensure_ascii=False,
                    )
                )
                if epochs_without_improvement >= args.early_stopping_patience:
                    stopped_early = True
                    _save_training_state(
                        state_path,
                        run_id=run_id,
                        model=model,
                        optimizer=optimizer,
                        experiment=experiment,
                        epoch_index=epoch + 1,
                        next_sample_index=0,
                        completed_epochs=completed_epochs,
                        epochs_without_improvement=epochs_without_improvement,
                        best_validation=best_validation,
                        stopped_early=stopped_early,
                    )
                    break
            _save_training_state(
                state_path,
                run_id=run_id,
                model=model,
                optimizer=optimizer,
                experiment=experiment,
                epoch_index=epoch + 1,
                next_sample_index=0,
                completed_epochs=completed_epochs,
                epochs_without_improvement=epochs_without_improvement,
                best_validation=best_validation,
                stopped_early=stopped_early,
            )
            start_sample_index = 0
        checkpoint = experiment.save_final(
            {
                "dataset_name": args.dataset,
                "split": args.split,
                "segmentation_method": args.method,
                "epochs_requested": args.epochs,
                "epochs_completed": completed_epochs,
                "steps_per_sample": args.steps_per_sample,
                "policy_version": policy_version,
                "stopped_early": stopped_early,
                "best_validation_qa_score": best_validation[0] if best_validation else None,
                "best_validation_normalized_cost": -best_validation[1] if best_validation else None,
            }
        )
        best_checkpoint = output_dir / "checkpoints" / "best.pt"
        if validation_segments_by_sample:
            if not best_checkpoint.is_file():
                raise RuntimeError(
                    "no validation checkpoint satisfied the configured budget; "
                    "the test partition must not be used for checkpoint selection"
                )
            selected_checkpoint = best_checkpoint
            selection_reason = "best_feasible_validation"
        else:
            # LoCoMo's predeclared outer CV has no validation partition. Its test fold
            # remains untouched; the final predeclared epoch checkpoint is selected.
            selected_checkpoint = checkpoint
            selection_reason = "final_predeclared_epoch_no_validation_partition"
        training_qa_usage = _summarize_qa_ledger(output_dir / "qa" / "ledger.sqlite3")
        validation_qa_usage = _summarize_qa_ledger(
            output_dir / "validation" / "evaluations.sqlite3"
        )
        update_experiment_manifest(
            manifest_path,
            status="complete",
            completed_at=datetime.now(timezone.utc).isoformat(),
            epochs_completed=completed_epochs,
            stopped_early=stopped_early,
            selected_checkpoint=str(selected_checkpoint.resolve()),
            selected_checkpoint_sha256=file_sha256(selected_checkpoint),
            selected_checkpoint_reason=selection_reason,
            final_checkpoint=str(checkpoint.resolve()),
            final_checkpoint_sha256=file_sha256(checkpoint),
            best_checkpoint=(str(best_checkpoint.resolve()) if best_checkpoint.is_file() else None),
            best_checkpoint_sha256=(file_sha256(best_checkpoint) if best_checkpoint.is_file() else None),
            best_validation_qa_score=best_validation[0] if best_validation else None,
            best_validation_normalized_cost=-best_validation[1] if best_validation else None,
            training_state=str(state_path.resolve()),
            training_state_sha256=file_sha256(state_path),
            training_qa_usage=training_qa_usage,
            validation_qa_usage=validation_qa_usage,
        )
        progress_metrics = {
            "epochs": completed_epochs,
            "stopped_early": stopped_early,
            "qa_tokens": training_qa_usage["total_tokens"],
            "qa_cost": training_qa_usage["total_cost"],
        }
        if latest is not None:
            progress_metrics.update(
                {
                    "last_qa": latest.qa_score,
                    "last_reward": latest.reward,
                }
            )
        training_progress.close(
            status="stopped_early" if stopped_early else "complete",
            metrics=progress_metrics,
        )
    except BaseException as exc:
        training_progress.close(
            status="failed", metrics={"error": type(exc).__name__}
        )
        update_experiment_manifest(
            manifest_path,
            status="failed",
            failed_at=datetime.now(timezone.utc).isoformat(),
            epochs_completed=completed_epochs,
            last_error=f"{type(exc).__name__}: {exc}",
        )
        raise
    finally:
        store.close()
    print(
        json.dumps(
            {
                "status": "complete",
                "checkpoint": str(selected_checkpoint),
                "final_checkpoint": str(checkpoint),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )


def _evaluate_validation_epoch(
    *,
    epoch: int,
    model,
    features_by_sample,
    segments_by_sample,
    examples,
    assembly_manager,
    evaluator,
    virtual_costs,
    normalizers,
    extraction_runs,
    policy_version: str,
    ledger: SqliteLedger,
) -> tuple[float, float]:
    total_correct = 0.0
    total_questions = 0
    normalized_costs = []
    progress = StageProgress(
        f"router validation epoch={epoch}",
        len(segments_by_sample),
        unit="samples",
    )
    try:
        for sample_id, segments in segments_by_sample.items():
            routes = model.route(features_by_sample[sample_id], deterministic=True)
            actions = [route.tier for route in routes]
            assembly = assembly_manager.create(
                dataset_name=segments[0].dataset_name,
                split=segments[0].split,
                sample_id=sample_id,
                segments=segments,
                actions=actions,
                probabilities=[route.probability for route in routes],
                episode_id=f"validation:{epoch:04d}:{sample_id}",
                policy_version=policy_version,
                router_type="embedding_mlp",
                candidate_extraction_run_id=extraction_runs[sample_id],
            )
            if assembly.status != "ready":
                raise RuntimeError(f"validation assembly failed for sample {sample_id}")
            try:
                questions = [pair.to_dict() for pair in examples[sample_id].qa_pairs]
                qa_score, _ = evaluator.evaluate_sample(
                    questions,
                    dataset_name=segments[0].dataset_name,
                    split=segments[0].split,
                    sample_id=sample_id,
                    assembly_id=assembly.assembly_id,
                    sample_metadata=examples[sample_id].metadata,
                )
                normalized_cost = normalizers[sample_id](
                    virtual_costs[sample_id](actions).total_cost
                )
                total_correct += qa_score * len(questions)
                total_questions += len(questions)
                normalized_costs.append(normalized_cost)
                ledger.append(
                    {
                        "epoch": epoch,
                        "sample_id": sample_id,
                        "assembly_id": assembly.assembly_id,
                        "qa_score": qa_score,
                        "normalized_cost": normalized_cost,
                        "num_questions": len(questions),
                        "route_decisions": actions,
                        "candidate_extraction_run_id": extraction_runs[sample_id],
                    }
                )
                progress.update(
                    item=sample_id,
                    metrics={
                        "qa": qa_score,
                        "questions": len(questions),
                        "norm_cost": normalized_cost,
                    },
                )
            finally:
                assembly_manager.cleanup(
                    assembly,
                    dataset_name=segments[0].dataset_name,
                    split=segments[0].split,
                )
    except BaseException:
        progress.close(status="failed")
        raise
    score = total_correct / total_questions
    cost = sum(normalized_costs) / len(normalized_costs)
    progress.close(metrics={"qa": score, "norm_cost": cost})
    return score, cost


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


def _require_training_api_keys(bundle) -> None:
    missing = [
        bundle.project.models[role].api_key_env
        for role in ("qa_reader", "judge_llm")
        if not bundle.project.models[role].resolved_api_key()
    ]
    if missing:
        raise RuntimeError(f"missing training API key environment variables: {missing}")


def _validate_segment_sets(segments_by_sample, dataset_name: str, split: str, method: str) -> None:
    versions = set()
    for sample_id, segments in segments_by_sample.items():
        for segment in segments:
            if (segment.dataset_name, segment.split, segment.sample_id, segment.segmentation_method) != (
                dataset_name,
                split,
                sample_id,
                method,
            ):
                raise ValueError(f"segment scope mismatch in sample {sample_id}")
            versions.add(segment.segmentation_version)
    if len(versions) != 1:
        raise ValueError(f"training samples mix segmentation versions: {sorted(versions)}")


def _validate_training_resume_manifest(path: Path, *, run_id: str, scope: dict) -> None:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("experiment_id") != run_id or manifest.get("run_type") != "router_training":
        raise ValueError("training resume manifest identity mismatch")
    if manifest.get("status") == "complete":
        raise ValueError("training run is already complete")
    actual = {key: manifest.get(key) for key in scope}
    if actual != scope:
        raise ValueError(f"training resume scope mismatch: expected={scope}, actual={actual}")


def _save_training_state(
    path: Path,
    *,
    run_id: str,
    model,
    optimizer,
    experiment,
    epoch_index: int,
    next_sample_index: int,
    completed_epochs: int,
    epochs_without_improvement: int,
    best_validation,
    stopped_early: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "schema_version": "router_training_state_v1",
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.optimizer.state_dict(),
            "lagrange_multiplier": optimizer.lagrange_multiplier,
            "global_step": experiment.global_step,
            "best_score": experiment.best_score,
            "epoch_index": epoch_index,
            "next_sample_index": next_sample_index,
            "completed_epochs": completed_epochs,
            "epochs_without_improvement": epochs_without_improvement,
            "best_validation": list(best_validation) if best_validation is not None else None,
            "stopped_early": stopped_early,
            "python_random_state": random.getstate(),
            "numpy_random_state": np.random.get_state(),
            "torch_random_state": torch.get_rng_state(),
            "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        },
        temporary,
    )
    temporary.replace(path)


def _load_training_state(path: Path, *, model, optimizer, experiment, expected_run_id: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"training state is missing: {path}")
    payload = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if payload.get("schema_version") != "router_training_state_v1":
        raise ValueError("unsupported training state schema")
    if payload.get("run_id") != expected_run_id:
        raise ValueError("training state run_id mismatch")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.optimizer.load_state_dict(payload["optimizer_state_dict"])
    optimizer.lagrange_multiplier = float(payload["lagrange_multiplier"])
    experiment.global_step = int(payload["global_step"])
    experiment.best_score = float(payload["best_score"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    cuda_state = payload.get("cuda_random_state")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)
    return payload


def _summarize_qa_ledger(path: Path) -> dict:
    rows = read_sqlite_ledger(path, "evaluations") if path.is_file() else []
    result = {"question_evaluations": len(rows)}
    for role in ("reader", "judge"):
        input_tokens = sum(int(row.get(f"{role}_input_tokens", 0)) for row in rows)
        output_tokens = sum(int(row.get(f"{role}_output_tokens", 0)) for row in rows)
        input_cost = sum(float(row.get(f"{role}_input_cost", 0.0)) for row in rows)
        output_cost = sum(float(row.get(f"{role}_output_cost", 0.0)) for row in rows)
        retries = sum(int(row.get(f"{role}_retry_count", 0)) for row in rows)
        result[role] = {
            "logical_api_calls": len(rows),
            "reported_retry_count": retries,
            "transport_attempts": len(rows) + retries,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
    result["total_tokens"] = result["reader"]["total_tokens"] + result["judge"]["total_tokens"]
    result["total_cost"] = result["reader"]["total_cost"] + result["judge"]["total_cost"]
    return result


if __name__ == "__main__":
    main()
